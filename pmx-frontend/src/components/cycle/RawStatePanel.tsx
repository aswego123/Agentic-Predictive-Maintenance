import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { JsonViewer } from '@/components/common/JsonViewer';
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from '@/components/ui/accordion';
import type { CycleSnapshot } from '@/types/domain';

export function RawStatePanel({ cycle }: { cycle: CycleSnapshot }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Raw state</CardTitle>
      </CardHeader>
      <CardContent>
        <Accordion type="single" collapsible>
          <AccordionItem value="raw" className="border-0">
            <AccordionTrigger className="text-xs">Show full JSON</AccordionTrigger>
            <AccordionContent>
              <JsonViewer data={cycle} maxHeight={480} />
            </AccordionContent>
          </AccordionItem>
        </Accordion>
      </CardContent>
    </Card>
  );
}
