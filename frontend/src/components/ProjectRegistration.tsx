import { useState } from 'react';
import { Plus, Loader2, AlertCircle, CheckCircle } from 'lucide-react';
import { useRegisterProject } from '../hooks/useProjects';
import { validateCid } from '../lib/utils';

interface ProjectRegistrationProps {
  onSuccess?: () => void;
}

export function ProjectRegistration({ onSuccess }: ProjectRegistrationProps) {
  const [cid, setCid] = useState('');
  const [endpoint, setEndpoint] = useState('');
  const [isOpen, setIsOpen] = useState(false);
  const registerProject = useRegisterProject();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!cid.trim()) return;
    
    if (!validateCid(cid.trim())) {
      alert('Please enter a valid IPFS CID');
      return;
    }

    if (!endpoint.trim()) {
      alert('Please enter a GraphQL endpoint URL');
      return;
    }

    try {
      await registerProject.mutateAsync({ 
        cid: cid.trim(),
        endpoint: endpoint.trim(),
      });
      setCid('');
      setEndpoint('');
      setIsOpen(false);
      onSuccess?.();
    } catch (error) {
      console.error('Failed to register project:', error);
    }
  };

  const isValidCid = cid.trim() && validateCid(cid.trim());
  const isValidEndpoint = endpoint.trim() && endpoint.trim().startsWith('http');
  const isFormValid = isValidCid && isValidEndpoint;

  return (
    <div className="space-y-4">
      {!isOpen ? (
        <button
          onClick={() => setIsOpen(true)}
          className="btn-primary w-full"
          disabled={registerProject.isPending}
        >
          <Plus className="w-4 h-4 mr-2" />
          Register New Project
        </button>
      ) : (
        <div className="card">
          <div className="card-header">
            <h3 className="card-title">Register SubQuery Project</h3>
            <p className="card-description">
              Enter the IPFS CID of your SubQuery project manifest and the GraphQL endpoint URL
            </p>
          </div>
          <div className="card-content">
            <form onSubmit={handleSubmit} className="space-y-4">
              <div className="space-y-2">
                <label htmlFor="cid" className="text-sm font-medium">
                  IPFS CID
                </label>
                <div className="relative">
                  <input
                    id="cid"
                    type="text"
                    value={cid}
                    onChange={(e) => setCid(e.target.value)}
                    placeholder="QmYourProjectCIDHere..."
                    className="input pr-10"
                    disabled={registerProject.isPending}
                  />
                  <div className="absolute inset-y-0 right-0 flex items-center pr-3">
                    {registerProject.isPending ? (
                      <Loader2 className="w-4 h-4 animate-spin text-muted-foreground" />
                    ) : cid.trim() ? (
                      isValidCid ? (
                        <CheckCircle className="w-4 h-4 text-green-500" />
                      ) : (
                        <AlertCircle className="w-4 h-4 text-red-500" />
                      )
                    ) : null}
                  </div>
                </div>
                {cid.trim() && !isValidCid && (
                  <p className="text-sm text-red-500">
                    Please enter a valid IPFS CID
                  </p>
                )}
              </div>

              <div className="space-y-2">
                <label htmlFor="endpoint" className="text-sm font-medium">
                  GraphQL Endpoint <span className="text-red-500">*</span>
                </label>
                <div className="relative">
                  <input
                    id="endpoint"
                    type="url"
                    value={endpoint}
                    onChange={(e) => setEndpoint(e.target.value)}
                    placeholder="https://api.subquery.network/sq/your-project..."
                    className="input pr-10"
                    disabled={registerProject.isPending}
                    required
                  />
                  <div className="absolute inset-y-0 right-0 flex items-center pr-3">
                    {registerProject.isPending ? (
                      <Loader2 className="w-4 h-4 animate-spin text-gray-400" />
                    ) : endpoint.trim() ? (
                      isValidEndpoint ? (
                        <CheckCircle className="w-4 h-4 text-green-500" />
                      ) : (
                        <AlertCircle className="w-4 h-4 text-red-500" />
                      )
                    ) : null}
                  </div>
                </div>
                {endpoint.trim() && !isValidEndpoint && (
                  <p className="text-sm text-red-500">
                    Please enter a valid HTTP/HTTPS URL
                  </p>
                )}
                <p className="text-xs text-gray-500">
                  The GraphQL endpoint URL for your SubQuery project
                </p>
              </div>

              {registerProject.error && (
                <div className="p-3 text-sm text-red-700 bg-red-50 border border-red-200 rounded-md">
                  <div className="flex items-center">
                    <AlertCircle className="w-4 h-4 mr-2" />
                    <span>
                      {registerProject.error instanceof Error 
                        ? registerProject.error.message 
                        : 'Failed to register project'}
                    </span>
                  </div>
                </div>
              )}

              <div className="flex space-x-3">
                <button
                  type="submit"
                  disabled={!isFormValid || registerProject.isPending}
                  className="btn-primary flex-1"
                >
                  {registerProject.isPending ? (
                    <>
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                      Registering...
                    </>
                  ) : (
                    'Register Project'
                  )}
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setIsOpen(false);
                    setCid('');
                    setEndpoint('');
                  }}
                  className="btn-outline"
                  disabled={registerProject.isPending}
                >
                  Cancel
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}