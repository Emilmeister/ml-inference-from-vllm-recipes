import { LockKeyhole, RotateCcw, SlidersHorizontal } from "lucide-react";

import type { RecipeDetail } from "../types";

type Props = {
  configuration: NonNullable<RecipeDetail["configuration"]>;
  onChange: (group: string, option: string, selected: boolean) => void;
  onReset: () => void;
};

export function RecipeConfigurationPicker({ configuration, onChange, onReset }: Props) {
  const groups = configuration.groups.filter((group) => group.options.length > 0);

  return (
    <div className="recipe-configurator">
      <div className="recipe-configurator__intro">
        <div>
          <p><SlidersHorizontal aria-hidden="true" size={18} />Конфигурация из vLLM Recipes</p>
          <span>Доступные комбинации пересчитываются сервером по опубликованным JSON.</span>
        </div>
        <button onClick={onReset} type="button"><RotateCcw aria-hidden="true" size={15} />По умолчанию</button>
      </div>

      <div className="recipe-configurator__groups">
        {groups.map((group) => {
          const groupId = `config-group-${group.key}`.replace(/[^a-zA-Z0-9_-]/g, "-");
          return (
          <div aria-labelledby={groupId} className="config-group" key={group.key} role="group">
            <p id={groupId}>{group.label}</p>
            <div className="config-options">
              {group.options.map((option) => {
                const unavailable = !option.available;
                const readOnly = group.control === "display";
                const explanation = unavailable
                  ? option.unavailable_reason
                  : option.selected ? option.description : undefined;
                const explanationId = explanation
                  ? `config-${group.key}-${option.key}`.replace(/[^a-zA-Z0-9_-]/g, "-")
                  : undefined;
                return (
                  <div className="config-option-wrap" key={option.key}>
                    <label
                      className={`config-option${option.selected ? " config-option--selected" : ""}${unavailable ? " config-option--disabled" : ""}`}
                      title={option.unavailable_reason || undefined}
                    >
                      <input
                        aria-describedby={explanationId}
                        checked={option.selected}
                        disabled={unavailable || readOnly}
                        name={`configuration-${group.key}`}
                        onChange={(event) => onChange(group.key, option.key, event.target.checked)}
                        type={group.control === "multiple" ? "checkbox" : "radio"}
                      />
                      {unavailable ? <LockKeyhole aria-hidden="true" size={13} /> : null}
                      <span>{option.label}</span>
                    </label>
                    {explanation ? (
                      <span className="config-option__explanation" id={explanationId}>
                        {explanation}
                      </span>
                    ) : null}
                  </div>
                );
              })}
            </div>
          </div>
          );
        })}
      </div>
    </div>
  );
}
