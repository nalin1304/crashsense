/** @type {import('tailwindcss').Config} */

/*
 * Color tokens below are tuned for WCAG 2.1 AA contrast ratios and validate
 * Requirements 25.2 (focus indicator ≥ 3:1) and 25.4 (body text ≥ 4.5:1,
 * non-text indicators ≥ 3:1). Each token includes the measured contrast
 * ratio against its intended background so future edits do not silently
 * regress accessibility.
 *
 *   Token             | Value     | Intended background      | Contrast
 *   ------------------+-----------+--------------------------+-----------
 *   crash.ink         | #0F172A   | white  (#FFFFFF)         | 17.85:1 (AAA)
 *   crash.slate       | #475569   | white  (#FFFFFF)         |  7.58:1 (AAA)
 *   crash.bodyDark    | #E2E8F0   | slate-950 (#020617)      | 16.36:1 (AAA)
 *   crash.mutedDark   | #94A3B8   | slate-950 (#020617)      |  7.87:1 (AAA)
 *   crash.focus       | #22D3EE   | slate-950 (#020617)      | 11.16:1 (≥3:1)
 *   crash.focusInk    | #0E7490   | white     (#FFFFFF)      |  5.36:1 (AA)
 */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        crash: {
          danger: '#ef4444',
          warn: '#f59e0b',
          ok: '#10b981',
          // Body text on light backgrounds (slate-900). 17.85:1 on white.
          ink: '#0F172A',
          // Muted text on light backgrounds (slate-600). 7.58:1 on white.
          slate: '#475569',
          // Body text on dark dashboard (slate-200). 16.36:1 on slate-950.
          bodyDark: '#E2E8F0',
          // Muted text on dark dashboard (slate-400). 7.87:1 on slate-950.
          mutedDark: '#94A3B8',
          // Focus ring on dark dashboard (cyan-400). 11.16:1 on slate-950.
          focus: '#22D3EE',
          // Focus ring on light surfaces (cyan-700). 5.36:1 on white.
          focusInk: '#0E7490',
        },
      },
      // Default focus ring colors so any element using `focus-visible:ring-2`
      // gets an accessible indicator without each call site picking a token.
      ringColor: {
        DEFAULT: '#22D3EE',
      },
      ringOffsetColor: {
        DEFAULT: '#020617',
      },
      // Body-text contrast tokens exposed as text colors so utilities like
      // `text-body-dark` / `text-body-ink` route to the audited values above.
      textColor: {
        'body-ink': '#0F172A',
        'body-muted': '#475569',
        'body-dark': '#E2E8F0',
        'body-muted-dark': '#94A3B8',
      },
      fontFamily: {
        sans: ['Inter', 'ui-sans-serif', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'ui-monospace', 'monospace'],
      },
      keyframes: {
        pulseDot: {
          '0%, 100%': { transform: 'scale(1)', opacity: '0.85' },
          '50%': { transform: 'scale(1.6)', opacity: '0.25' },
        },
        wavefront: {
          '0%': { transform: 'scale(0.1)', opacity: '0.9' },
          '100%': { transform: 'scale(2.6)', opacity: '0' },
        },
      },
      animation: {
        pulseDot: 'pulseDot 1.6s ease-in-out infinite',
        wavefront: 'wavefront 3s ease-out forwards',
      },
    },
  },
  plugins: [],
};
