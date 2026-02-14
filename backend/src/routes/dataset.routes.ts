import express, { Request, Response } from 'express';
import { authenticateToken, requireAdmin } from '../auth/middleware';
import { PrismaClient } from '@prisma/client';
import multer from 'multer';
import path from 'path';
import fs from 'fs';
import axios from 'axios';

const router = express.Router();
const prisma = new PrismaClient();
const AI_SERVICE_URL = process.env.AI_SERVICE_URL || 'http://localhost:8000';

// Configure Multer for file uploads
const uploadDir = path.join(__dirname, '../../uploads/datasets');
if (!fs.existsSync(uploadDir)) {
    fs.mkdirSync(uploadDir, { recursive: true });
}

const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        cb(null, uploadDir);
    },
    filename: (req, file, cb) => {
        const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
        cb(null, file.fieldname + '-' + uniqueSuffix + path.extname(file.originalname));
    }
});

const upload = multer({
    storage: storage,
    limits: { fileSize: 100 * 1024 * 1024 }, // 100MB limit
    fileFilter: (req, file, cb) => {
        if (file.mimetype === 'text/csv' || file.mimetype === 'application/vnd.ms-excel' || file.originalname.endsWith('.csv')) {
            cb(null, true);
        } else {
            cb(new Error('Only CSV files are allowed!'));
        }
    }
});

// GET /api/admin/datasets - List all datasets
router.get('/', authenticateToken, requireAdmin, async (req: Request, res: Response) => {
    try {
        const datasets = await prisma.dataset.findMany({
            orderBy: { uploadedAt: 'desc' },
            include: { models: true }
        });
        res.json(datasets);
    } catch (error: any) {
        res.status(500).json({ error: "Failed to fetch datasets", details: error.message });
    }
});

// POST /api/admin/datasets/upload - Upload new dataset
router.post('/upload', authenticateToken, requireAdmin, upload.single('file'), async (req: Request, res: Response) => {
    try {
        if (!req.file) {
            return res.status(400).json({ error: "No file uploaded" });
        }

        // Basic CSV analysis (read first line for columns)
        const filePath = req.file.path;
        const fileContent = fs.readFileSync(filePath, 'utf-8');
        const lines = fileContent.split('\n');
        const rowCount = lines.length - 1; // Approximate
        const columns = lines[0].split(',').map(c => c.trim());

        const dataset = await prisma.dataset.create({
            data: {
                name: req.body.name || req.file.originalname,
                path: filePath,
                size: req.file.size,
                rowCount: Math.max(0, rowCount),
                columns: JSON.stringify(columns),
                status: 'uploaded'
            }
        });

        res.json(dataset);

    } catch (error: any) {
        res.status(500).json({ error: "Failed to upload dataset", details: error.message });
    }
});

// POST /api/admin/datasets/kaggle - Import from Kaggle
router.post('/kaggle', authenticateToken, requireAdmin, async (req: Request, res: Response) => {
    try {
        const { handle } = req.body;
        if (!handle) {
            return res.status(400).json({ error: "Kaggle dataset handle is required" });
        }

        // Trigger AI Service to download
        const response = await axios.post(`${AI_SERVICE_URL}/download-kaggle`, { handle });
        const metadata = response.data;

        // Create Dataset record
        const dataset = await prisma.dataset.create({
            data: {
                name: metadata.name,
                path: metadata.path,
                size: metadata.size,
                rowCount: metadata.rowCount,
                columns: JSON.stringify(metadata.columns),
                status: 'uploaded' // Ready for validation
            }
        });

        res.json(dataset);

    } catch (error: any) {
        console.error("Kaggle import error:", error.response?.data || error.message);
        res.status(500).json({
            error: "Failed to import from Kaggle",
            details: error.response?.data?.detail || error.message
        });
    }
});

// POST /api/admin/datasets/:id/validate - Trigger validation/profiling
router.post('/:id/validate', authenticateToken, requireAdmin, async (req: Request, res: Response) => {
    try {
        const id = parseInt(req.params.id);
        const dataset = await prisma.dataset.findUnique({ where: { id } });

        if (!dataset) {
            return res.status(404).json({ error: "Dataset not found" });
        }

        // Update status
        await prisma.dataset.update({
            where: { id },
            data: { status: 'processing' }
        });

        // Trigger AI Service
        try {
            const response = await axios.post(`${AI_SERVICE_URL}/validate`, {
                dataset_path: dataset.path
            });

            // Update with results (assuming response contains report path or success status)
            await prisma.dataset.update({
                where: { id },
                data: { status: 'ready' }
            });

            res.json({ message: "Validation started", ai_response: response.data });

        } catch (aiError: any) {
            await prisma.dataset.update({
                where: { id },
                data: { status: 'error' }
            });
            throw new Error(`AI Service failed: ${aiError.message}`);
        }

    } catch (error: any) {
        res.status(500).json({ error: "Failed to validate dataset", details: error.message });
    }
});

// GET /api/admin/datasets/:id/report - Get profiling report (placeholder for now)
router.get('/:id/report', authenticateToken, requireAdmin, async (req: Request, res: Response) => {
    // In a real implementation, this would serve the HTML report generated by pandas-profiling
    // For now, we'll return the dataset info
    try {
        const id = parseInt(req.params.id);
        const dataset = await prisma.dataset.findUnique({ where: { id } });
        if (!dataset) return res.status(404).json({ error: "Dataset not found" });

        res.json({ message: "Report viewing not fully implemented yet", dataset });
    } catch (error: any) {
        res.status(500).json({ error: "Failed to fetch report", details: error.message });
    }
});

export default router;
