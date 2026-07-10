import { Download } from 'lucide-react';

import { Button } from '@/components/ui/button';
import type { CycleSnapshot } from '@/types/domain';

function download(filename: string, mime: string, content: string) {
  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

export function DownloadReportButton({ cycle }: { cycle: CycleSnapshot }) {
  const exportJson = () => {
    download(`${cycle.cycle_id}.json`, 'application/json', JSON.stringify(cycle, null, 2));
  };
  return (
    <Button variant="outline" size="sm" onClick={exportJson} title="Download full cycle snapshot as JSON">
      <Download className="mr-1 h-4 w-4" />
      Export JSON
    </Button>
  );
}
