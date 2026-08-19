"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { getHealth } from "@/lib/api";

export default function Navbar() {
  const [apiOk, setApiOk] = useState<boolean | null>(null);

  useEffect(() => {
    let cancelled = false;
    getHealth()
      .then((h) => !cancelled && setApiOk(h.artifacts_loaded))
      .catch(() => !cancelled && setApiOk(false));
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <nav className="border-b border-border bg-bg/80 backdrop-blur sticky top-0 z-10">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 h-14 flex items-center justify-between">
        <Link href="/" className="font-mono font-semibold text-accent">
          AI-NIDS
        </Link>
        <div className="flex items-center gap-6 text-sm">
          <Link href="/" className="hover:text-accent">
            Home
          </Link>
          <Link href="/analyzer" className="hover:text-accent">
            Analyzer
          </Link>
          <Link href="/research" className="hover:text-accent">
            Research
          </Link>
          <span
            className={`flex items-center gap-2 font-mono text-xs ${
              apiOk === null
                ? "text-muted"
                : apiOk
                ? "text-accent2"
                : "text-danger"
            }`}
            title={
              apiOk === null
                ? "Checking backend…"
                : apiOk
                ? "Backend reachable"
                : "Backend unreachable"
            }
          >
            <span
              className={`w-2 h-2 rounded-full ${
                apiOk === null
                  ? "bg-muted"
                  : apiOk
                  ? "bg-accent2 animate-pulse"
                  : "bg-danger"
              }`}
            />
            API {apiOk === null ? "…" : apiOk ? "Online" : "Offline"}
          </span>
        </div>
      </div>
    </nav>
  );
}
