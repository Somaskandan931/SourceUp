export const resolveSupplierName = (s) => {
  if (!s) return null;
  const raw =
    s.supplier ||
    s['supplier name'] || s['Supplier Name'] ||
    s['company name']  || s['Company Name'] ||
    s.company || s.name || s.brand || s.manufacturer;
  if (!raw || /^unknown/i.test(raw.trim())) return null;
  return raw.trim();
};

export const resolveProductName = (s, fallback = '') => {
  if (!s) return fallback;
  return (
    s.product ||
    s['product name'] || s['Product Name'] ||
    s['Product'] ||
    s.item || s.description || fallback
  ).trim();
};
