'use client';

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';

export default function DashboardPage() {
    const router = useRouter();

    useEffect(() => {
        // Redirect to first available tab
        router.push('/dashboard/diagnostics');
    }, [router]);

    return null;
}
