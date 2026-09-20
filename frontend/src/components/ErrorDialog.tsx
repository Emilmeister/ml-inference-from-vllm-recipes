import { TriangleAlert, X } from "lucide-react";
import { useEffect, useId, useRef } from "react";

type ErrorDialogProps = {
  message: string | null;
  onClose: () => void;
};

export function ErrorDialog({ message, onClose }: ErrorDialogProps) {
  const dialogRef = useRef<HTMLDialogElement>(null);
  const titleId = useId();

  useEffect(() => {
    const dialog = dialogRef.current;
    if (!dialog) return;
    if (message && !dialog.open) dialog.showModal();
    if (!message && dialog.open) dialog.close();
  }, [message]);

  return (
    <dialog
      aria-labelledby={titleId}
      className="error-dialog"
      onCancel={onClose}
      ref={dialogRef}
    >
      <button
        aria-label="Закрыть сообщение об ошибке"
        className="error-dialog__close"
        onClick={onClose}
        title="Закрыть"
        type="button"
      >
        <X aria-hidden="true" size={18} />
      </button>
      <TriangleAlert aria-hidden="true" className="error-dialog__icon" size={28} />
      <p className="kicker">Создание не выполнено</p>
      <h2 id={titleId}>Cloud.ru вернул ошибку</h2>
      <p>{message}</p>
      <button className="primary-button" onClick={onClose} type="button">Закрыть</button>
    </dialog>
  );
}
