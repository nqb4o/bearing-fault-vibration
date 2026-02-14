import express, { Request, Response } from 'express';
import { authenticateToken } from '../auth/middleware';
import axios from 'axios';
import multer from 'multer';
import FormData from 'form-data';
import fs from 'fs';
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

const router = express.Router();
const upload = multer({ dest: 'uploads/' }); // Temporary storage
const AI_SERVICE_URL = process.env.AI_SERVICE_URL || 'http://localhost:8000';

// Inference Endpoint
router.post('/predict', authenticateToken, upload.single('file'), async (req: Request, res: Response) => {
    const file = (req as any).file;
    const modelId = req.body.modelId ? parseInt(req.body.modelId) : null;

    if (!file) {
        return res.status(400).json({ error: "No file uploaded" });
    }

    // Dynamic Model Loading
    if (modelId) {
        try {
            const model = await prisma.trainedModel.findUnique({ where: { id: modelId } });
            if (model) {
                // Determine base path (remove extension)
                let modelPathPrefix = model.path;
                if (modelPathPrefix.endsWith('.h5')) {
                    modelPathPrefix = modelPathPrefix.slice(0, -3);
                }

                await axios.post(`${AI_SERVICE_URL}/load-model`, {
                    model_path: modelPathPrefix
                });
            }
        } catch (loadError) {
            console.error("Failed to load model:", loadError);
            return res.status(500).json({ error: "Failed to load selected model" });
        }
    }

    try {
        const formData = new FormData();
        formData.append('file', fs.createReadStream(file.path));

        const response = await axios.post(`${AI_SERVICE_URL}/inference`, formData, {
            headers: {
                ...formData.getHeaders()
            }
        });

        // Cleanup temp file
        fs.unlinkSync(file.path);

        res.json(response.data);
    } catch (error: any) {
        // Cleanup temp file on error too
        if (file) fs.unlinkSync(file.path);

        const status = error.response?.status || 500;
        const message = error.response?.data?.detail || error.response?.data?.error || "Inference failed";

        res.status(status).json({ error: message, details: error.message });
    }
});

export default router;
