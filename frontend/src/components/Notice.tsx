import { CircleAlert, Info, TriangleAlert } from "lucide-react";
import type { ReactNode } from "react";

type NoticeProps = {
  children: ReactNode;
  id?: string;
  live?: boolean;
  title?: string;
  tone?: "info" | "warning" | "error";
};

export function Notice({ children, id, live = false, title, tone = "info" }: NoticeProps) {
  const Icon = tone === "warning" ? TriangleAlert : tone === "error" ? CircleAlert : Info;

  return (
    <div
      aria-atomic={live || undefined}
      aria-live={live ? "polite" : undefined}
      className={`notice notice--${tone}`}
      id={id}
      role={tone === "error" ? "alert" : "note"}
    >
      <Icon aria-hidden="true" size={19} strokeWidth={2.2} />
      <div>
        {title ? <strong>{title}</strong> : null}
        <div className="notice__copy">{children}</div>
      </div>
    </div>
  );
}
