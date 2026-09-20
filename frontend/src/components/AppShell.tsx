import { BookOpen, Cloud, RadioTower, Settings, Trash2 } from "lucide-react";
import { NavLink, Outlet, useNavigate } from "react-router-dom";

import { clearCredentials } from "../credentials";

const navigation = [
  { to: "/inferences", label: "Мои инференсы", icon: RadioTower },
  { to: "/catalog", label: "Каталог", icon: BookOpen },
  { to: "/setup", label: "Настройки", icon: Settings },
];

export function AppShell({ onCredentialsCleared }: { onCredentialsCleared?: () => void }) {
  const navigate = useNavigate();

  function clearSettings() {
    if (!window.confirm("Удалить сохранённые настройки Cloud.ru из этого браузера?")) return;
    clearCredentials();
    onCredentialsCleared?.();
    navigate("/setup", { replace: true });
  }

  return (
    <div className="app-shell">
      <a className="skip-link" href="#main-content">
        Перейти к содержимому
      </a>
      <aside className="sidebar">
        <div className="brand">
          <span className="brand__mark" aria-hidden="true">
            <Cloud size={22} strokeWidth={2.4} />
          </span>
          <span>
            <strong>Recipe Relay</strong>
            <small>vLLM → Cloud.ru</small>
          </span>
        </div>

        <nav className="main-nav" aria-label="Основная навигация">
          {navigation.map(({ to, label, icon: Icon }) => (
            <NavLink
              className={({ isActive }) => `nav-link${isActive ? " nav-link--active" : ""}`}
              key={to}
              to={to}
            >
              <Icon aria-hidden="true" size={19} strokeWidth={2.1} />
              <span>{label}</span>
            </NavLink>
          ))}
        </nav>

        <div className="sidebar__footer">
          <div className="sidebar__note">
            <span className="status-dot" aria-hidden="true" />
            <span>Локальная консоль</span>
          </div>
          <button className="clear-settings" onClick={clearSettings} type="button">
            <Trash2 aria-hidden="true" size={16} />
            Очистить настройки
          </button>
        </div>
      </aside>

      <main className="shell-content" id="main-content">
        <Outlet />
      </main>
    </div>
  );
}
