import { Navigate, Route, Routes } from "react-router-dom";
import { useCallback, useEffect, useState } from "react";

import { AppShell } from "./components/AppShell";
import { loadCredentials } from "./credentials";
import { CatalogPage } from "./pages/CatalogPage";
import { DeployPage } from "./pages/DeployPage";
import { InferencesPage } from "./pages/InferencesPage";
import { RecipePage } from "./pages/RecipePage";
import { SetupPage } from "./pages/SetupPage";
// @ts-expect-error Vite handles CSS side-effect imports at build time.
import "./styles.css";

function ProtectedShell({
  hasCredentials,
  onCredentialsCleared,
}: {
  hasCredentials: boolean;
  onCredentialsCleared: () => void;
}) {
  return hasCredentials
    ? <AppShell onCredentialsCleared={onCredentialsCleared} />
    : <Navigate replace to="/setup" />;
}

function HomeRedirect({ hasCredentials }: { hasCredentials: boolean }) {
  return <Navigate replace to={hasCredentials ? "/inferences" : "/setup"} />;
}

export function App() {
  const [expiresAt, setExpiresAt] = useState<number | null>(
    () => loadCredentials()?.expiresAt ?? null,
  );
  const refreshCredentials = useCallback(() => {
    setExpiresAt(loadCredentials()?.expiresAt ?? null);
  }, []);

  useEffect(() => {
    window.addEventListener("storage", refreshCredentials);
    if (expiresAt === null) {
      return () => window.removeEventListener("storage", refreshCredentials);
    }
    const delay = Math.min(Math.max(0, expiresAt - Date.now()), 2_147_483_647);
    const timer = window.setTimeout(refreshCredentials, delay);
    return () => {
      window.clearTimeout(timer);
      window.removeEventListener("storage", refreshCredentials);
    };
  }, [expiresAt, refreshCredentials]);

  const hasCredentials = expiresAt !== null;

  return (
    <Routes>
      <Route element={<SetupPage onSaved={refreshCredentials} />} path="/setup" />
      <Route
        element={
          <ProtectedShell
            hasCredentials={hasCredentials}
            onCredentialsCleared={refreshCredentials}
          />
        }
      >
        <Route element={<InferencesPage />} path="/inferences" />
        <Route element={<CatalogPage />} path="/catalog" />
        <Route element={<RecipePage />} path="/catalog/:organization/:model" />
        <Route element={<DeployPage />} path="/deploy" />
      </Route>
      <Route element={<HomeRedirect hasCredentials={hasCredentials} />} path="*" />
    </Routes>
  );
}
