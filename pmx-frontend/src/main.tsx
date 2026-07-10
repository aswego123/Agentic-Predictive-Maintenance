import React from 'react';
import ReactDOM from 'react-dom/client';
import { RouterProvider } from 'react-router-dom';
import { QueryClientProvider } from '@tanstack/react-query';
import { ReactQueryDevtools } from '@tanstack/react-query-devtools';
import { Toaster } from 'sonner';

import { router } from '@/routes/router';
import { queryClient } from '@/services/queryClient';
import { useAppStore } from '@/store';
import '@/styles/globals.css';

// Apply persisted theme before first paint.
const initialTheme = useAppStore.getState().session.theme;
document.documentElement.classList.toggle('dark', initialTheme === 'dark');

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <RouterProvider router={router} />
      <Toaster richColors position="top-right" />
      <ReactQueryDevtools initialIsOpen={false} />
    </QueryClientProvider>
  </React.StrictMode>,
);
