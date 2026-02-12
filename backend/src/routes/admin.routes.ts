import express, { Request, Response } from 'express';
import { authenticateToken, requireAdmin } from '../auth/middleware';
import axios from 'axios';

const router = express.Router();
const AI_SERVICE_URL = process.env.AI_SERVICE_URL || 'http://localhost:8000';

// Trigger Training
router.post('/train', authenticateToken, requireAdmin, async (req: Request, res: Response) => {
    try {
        const response = await axios.post(`${AI_SERVICE_URL}/train`, req.body);
        res.json(response.data);
    } catch (error: any) {
        res.status(500).json({ error: "Failed to trigger training", details: error.message });
    }
});

// Get Last Training History
router.get('/train/history', authenticateToken, requireAdmin, async (req: Request, res: Response) => {
    try {
        const response = await axios.get(`${AI_SERVICE_URL}/train/history`);
        res.json(response.data);
    } catch (error: any) {
        res.status(500).json({ error: "Failed to fetch training history", details: error.message });
    }
});

// SSE Training Stream - proxies real-time progress from AI service
router.post('/train/stream', authenticateToken, requireAdmin, async (req: Request, res: Response) => {
    try {
        const response = await axios.post(`${AI_SERVICE_URL}/train/stream`, req.body, {
            responseType: 'stream',
            headers: { 'Accept': 'text/event-stream', 'Content-Type': 'application/json' },
            timeout: 0, // No timeout for long training
        });

        res.setHeader('Content-Type', 'text/event-stream');
        res.setHeader('Cache-Control', 'no-cache');
        res.setHeader('Connection', 'keep-alive');
        res.setHeader('X-Accel-Buffering', 'no');

        response.data.pipe(res);
    } catch (error: any) {
        res.status(500).json({ error: "Training stream failed", details: error.message });
    }
});

export default router;
