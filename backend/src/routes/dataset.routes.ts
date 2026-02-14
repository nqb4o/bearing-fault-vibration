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

// POST /api/admin/datasets/upload - Upload new dataset (Multiple files support)
router.post('/upload', authenticateToken, requireAdmin, upload.array('files'), async (req: Request, res: Response) => {
    try {
        if (!req.files || (req.files as Express.Multer.File[]).length === 0) {
            return res.status(400).json({ error: "No files uploaded" });
        }

        const files = req.files as Express.Multer.File[];

        // Create a specific folder for this dataset upload to keep things organized
        // Use the timestamp-random suffix from the first file's generation or a new one
        const datasetId = Date.now() + '-' + Math.round(Math.random() * 1E5);
        const datasetPath = path.join(uploadDir, datasetId);

        if (!fs.existsSync(datasetPath)) {
            fs.mkdirSync(datasetPath, { recursive: true });
        }

        // Move files to this new folder
        let totalSize = 0;
        let rowCount = 0;
        let columns: string[] = [];
        let totalFiles = 0;

        for (const file of files) {
            const oldPath = file.path;
            const newPath = path.join(datasetPath, file.originalname);
            fs.renameSync(oldPath, newPath);
            totalSize += file.size;
            totalFiles++;

            // Analyze the first file found to get columns
            if (columns.length === 0) {
                try {
                    const content = fs.readFileSync(newPath, 'utf-8');
                    const lines = content.split('\n');
                    if (lines.length > 0) {
                        columns = lines[0].split(',').map(c => c.trim());
                        // Simple row count estimation from one file is not accurate for multi-file
                        // We will just store 0 or update later via profiling
                        rowCount = lines.length - 1;
                    }
                } catch (e) { }
            }
        }

        // Use the name from the request or the folder ID
        const name = req.body.name || `Dataset ${datasetId}`;

        const dataset = await prisma.dataset.create({
            data: {
                name: name,
                path: datasetPath, // Point to the DIRECTORY, not a file
                size: totalSize,
                rowCount: rowCount, // This is just an estimate from one file, real count comes later
                columns: JSON.stringify(columns),
                status: 'uploaded'
            }
        });

        res.json(dataset);

    } catch (error: any) {
        // Cleanup on error (optional but good practice)
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
                data: {
                    status: 'ready',
                    validationResult: response.data // Save the full AI response (valid, checks, report_path)
                }
            });

            res.json({ message: "Validation completed", ai_response: response.data });

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
