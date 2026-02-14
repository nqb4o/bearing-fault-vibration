import express, { Request, Response } from 'express';
import { authenticateToken } from '../auth/middleware';
import { PrismaClient } from '@prisma/client';

const router = express.Router();
const prisma = new PrismaClient();
const API_BASE_URL = process.env.API_BASE_URL || 'http://localhost:4000/api';

// GET /api/models - List all trained models (Accessible by Users)
router.get('/', authenticateToken, async (req: Request, res: Response) => {
    try {
        const models = await prisma.trainedModel.findMany({
            orderBy: { createdAt: 'desc' },
            include: { dataset: true }
        });
        res.json(models);
    } catch (error: any) {
        res.status(500).json({ error: "Failed to fetch models", details: error.message });
    }
});

// GET /api/models/:id/snippet - Get code snippet for inference
router.get('/:id/snippet', authenticateToken, async (req: Request, res: Response) => {
    try {
        const id = parseInt(req.params.id);
        const model = await prisma.trainedModel.findUnique({ where: { id } });

        if (!model) {
            return res.status(404).json({ error: "Model not found" });
        }

        const pythonSnippet = `
import requests

url = "${API_BASE_URL}/inference"
files = {'file': open('sensor_data.csv', 'rb')}
data = {'model_id': ${id}}
headers = {'Authorization': 'Bearer YOUR_TOKEN'}

response = requests.post(url, files=files, data=data, headers=headers)
print(response.json())
`;

        const curlSnippet = `
curl -X POST "${API_BASE_URL}/inference" \\
     -H "Authorization: Bearer YOUR_TOKEN" \\
     -F "file=@sensor_data.csv" \\
     -F "model_id=${id}"
`;

        res.json({
            python: pythonSnippet.trim(),
            curl: curlSnippet.trim()
        });

    } catch (error: any) {
        res.status(500).json({ error: "Failed to generate snippet", details: error.message });
    }
});

// POST /api/models - Register a newly trained model
router.post('/', authenticateToken, async (req: Request, res: Response) => {
    try {
        const { name, version, accuracy, datasetId, path } = req.body;

        if (!name || accuracy === undefined || !path) {
            return res.status(400).json({ error: "Missing required fields: name, accuracy, path" });
        }

        const model = await prisma.trainedModel.create({
            data: {
                name,
                version: version || '1.0',
                accuracy: parseFloat(accuracy),
                path,
                datasetId: datasetId ? parseInt(datasetId) : undefined
            }
        });

        res.status(201).json(model);
    } catch (error: any) {
        console.error("Model registration failed:", error);
        // Debugging: Log to file
        const fs = require('fs');
        fs.appendFileSync('backend_errors.log', `${new Date().toISOString()} - Model Reg Error: ${error.message}\n`);
        res.status(500).json({ error: "Failed to register model", details: error.message });
    }
});

export default router;
