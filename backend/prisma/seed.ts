import { PrismaClient } from '@prisma/client';
import bcrypt from 'bcrypt';

const prisma = new PrismaClient();

async function main() {
  // Hash password 'demo123'
  const hashedPassword = await bcrypt.hash('demo123', 10);

  // Admin Account
  const admin = await prisma.user.upsert({
    where: { email: 'admin@demo.com' },
    update: {},
    create: {
      email: 'admin@demo.com',
      password: hashedPassword,
      role: 'ADMIN',
    },
  });

  // User Account
  const user = await prisma.user.upsert({
    where: { email: 'user@demo.com' },
    update: {},
    create: {
      email: 'user@demo.com',
      password: hashedPassword,
      role: 'USER',
    },
  });

  console.log('Created Demo Accounts:');
  console.log('Admin: admin@demo.com / demo123');
  console.log('User:  user@demo.com  / demo123');
}

main()
  .then(async () => {
    await prisma.$disconnect();
  })
  .catch(async (e) => {
    console.error(e);
    await prisma.$disconnect();
    process.exit(1);
  });
