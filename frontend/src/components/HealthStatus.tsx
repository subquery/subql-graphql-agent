import { useHealth } from '../hooks/useProjects';
import { Activity, AlertTriangle, Loader2 } from 'lucide-react';

export function HealthStatus() {
  const { data: health, isLoading, error } = useHealth();

  if (isLoading) {
    return (
      <div className="flex items-center space-x-2 text-gray-500">
        <Loader2 className="w-4 h-4 animate-spin" />
        <span className="text-sm">Connecting...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center space-x-2 text-red-500">
        <AlertTriangle className="w-4 h-4" />
        <span className="text-sm">Backend offline</span>
      </div>
    );
  }

  if (health) {
    return (
      <div className="flex items-center space-x-4">
        <div className="flex items-center space-x-2 text-green-600">
          <Activity className="w-4 h-4" />
          <span className="text-sm">Online</span>
        </div>
        <div className="text-sm text-gray-500">
          {health.projects_count} projects • {health.cached_agents} cached
        </div>
      </div>
    );
  }

  return null;
}