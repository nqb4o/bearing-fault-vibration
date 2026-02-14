
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

async function checkModels() {
    try {
        const models = await prisma.trainedModel.findMany();
        console.log("Found Models:", models);
    } catch (e) {
        console.error("Error fetching models:", e);
    } finally {
        await prisma.$disconnect();
    }
}

checkModels();
