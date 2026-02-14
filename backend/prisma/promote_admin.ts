
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

async function main() {
    const email = 'admin@example.com';

    console.log(`Checking user: ${email}...`);

    const user = await prisma.user.findUnique({
        where: { email },
    });

    if (!user) {
        console.error(`User ${email} not found!`);
        return;
    }

    console.log(`Current role: ${user.role}`);

    if (user.role !== 'ADMIN') {
        const updated = await prisma.user.update({
            where: { email },
            data: { role: 'ADMIN' },
        });
        console.log(`User promoted to ADMIN. New role: ${updated.role}`);
    } else {
        console.log('User is already ADMIN.');
    }
}

main()
    .catch((e) => {
        console.error(e);
        process.exit(1);
    })
    .finally(async () => {
        await prisma.$disconnect();
    });
