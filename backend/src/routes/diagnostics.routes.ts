import express, { Request, Response } from 'express';
import { authenticateToken } from '../auth/middleware';
import axios from 'axios';
import multer from 'multer';
import FormData from 'form-data';
import fs from 'fs';

const router = express.Router();
const upload = multer({ dest: 'uploads/' }); // Temporary storage
const AI_SERVICE_URL = process.env.AI_SERVICE_URL || 'http://localhost:8000';

// Inference Endpoint
router.post('/predict', authenticateToken, upload.single('file'), async (req: Request, res: Response) => {
    const file = (req as any).file;
    if (!file) {
        return res.status(400).json({ error: "No file uploaded" });
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
        res.status(500).json({ error: "Inference failed", details: error.message });
    }
});

export default router;
