// components/Spinner.jsx
export default function Spinner({ size = 16, color = '#fff' }) {
  return (
    <svg className="spin" width={size} height={size} viewBox="0 0 24 24" fill="none">
      <circle cx="12" cy="12" r="10" stroke={color + '40'} strokeWidth="3"/>
      <path d="M12 2a10 10 0 0 1 10 10" stroke={color} strokeWidth="3" strokeLinecap="round"/>
    </svg>
  );
}
