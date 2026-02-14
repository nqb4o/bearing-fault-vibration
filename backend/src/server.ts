import express, { Request, Response } from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import authRoutes from './routes/auth.routes';
import adminRoutes from './routes/admin.routes';
import diagnosticsRoutes from './routes/diagnostics.routes';
import datasetRoutes from './routes/dataset.routes';
import modelRoutes from './routes/model.routes';

dotenv.config();

const app = express();
const PORT = process.env.PORT || 4000;

app.use(cors());
app.use(express.json());

// Routes
app.use('/api/auth', authRoutes);
app.use('/api/admin', adminRoutes);
app.use('/api/diagnostics', diagnosticsRoutes);
app.use('/api/admin/datasets', datasetRoutes);
app.use('/api/models', modelRoutes);

app.get('/', (req: Request, res: Response) => {
    res.send('Bearing Fault Diagnostic Backend Running');
});

app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});
