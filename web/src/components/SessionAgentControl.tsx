import type { AgentInfo } from "../api/types";

interface Props {
  agents: AgentInfo[];
  agentName: string;
  onSelectAgent: (name: string) => void;
}

export function SessionAgentControl({
  agents,
  agentName,
  onSelectAgent,
}: Props) {
  const label = agents[0]?.name || agentName || "default";
  if (agents.length <= 1) {
    return (
      <span className="model-badge agent-badge" title="当前 Agent">
        {label}
      </span>
    );
  }
  return (
    <select
      className="model-badge agent-badge"
      aria-label="选择 Agent"
      value={agentName}
      onChange={(ev) => onSelectAgent(ev.target.value)}
    >
      {agents.map((item) => (
        <option key={item.name} value={item.name}>
          {item.name}
        </option>
      ))}
    </select>
  );
}
