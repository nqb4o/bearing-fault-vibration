import Link from 'next/link'
import { ArrowRight, ShieldCheck } from 'lucide-react'

export default function Home() {
    return (
        <main className="flex min-h-screen flex-col items-center justify-center p-24 bg-white text-black">
            <div className="z-10 max-w-5xl w-full items-center justify-between font-mono text-sm lg:flex">
                <p className="fixed left-0 top-0 flex w-full justify-center border-b border-gray-300 bg-gradient-to-b from-zinc-200 pb-6 pt-8 backdrop-blur-2xl lg:static lg:w-auto  lg:rounded-xl lg:border lg:bg-gray-200 lg:p-4 lg:dark:bg-zinc-800/30">
                    BearingGuard AI &nbsp;
                    <code className="font-mono font-bold">v2.0</code>
                </p>
            </div>

            <div className="relative flex place-items-center flex-col gap-6">
                <ShieldCheck size={100} className="text-blue-500 mb-4" />
                <h1 className="text-6xl font-bold tracking-tighter text-transparent bg-clip-text bg-gradient-to-r from-blue-600 to-indigo-600">
                    BearingGuard
                </h1>
                <p className="text-xl text-gray-600 max-w-lg text-center">
                    Industrial-grade bearing fault detection powered by Advanced AI.
                    Upload sensor data, get instant diagnostics.
                </p>

                <div className="flex gap-4 mt-8">
                    <Link href="/login" className="bg-blue-600 text-white px-8 py-3 rounded-full font-semibold hover:bg-blue-700 transition flex items-center gap-2">
                        Get Started <ArrowRight size={18} />
                    </Link>
                    <Link href="/signup" className="border border-gray-300 text-gray-700 px-8 py-3 rounded-full font-semibold hover:bg-gray-50 transition">
                        Create Account
                    </Link>
                </div>
            </div>
        </main>
    )
}
