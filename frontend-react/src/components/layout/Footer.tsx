import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';

const links = {
  company: [
    { label: 'About us', href: '/coming-soon' },
    { label: 'Careers', href: '/coming-soon' },
    { label: 'Press', href: '/coming-soon' },
    { label: 'Blog', href: '/coming-soon' },
  ],
  support: [
    { label: 'Help Center', href: '/coming-soon' },
    { label: 'Contact', href: '/coming-soon' },
    { label: 'Partner with us', href: '/coming-soon' },
  ],
  legal: [
    { label: 'Terms', href: '/coming-soon' },
    { label: 'Privacy', href: '/coming-soon' },
    { label: 'Cookies', href: '/coming-soon' },
  ],
};

export function Footer() {
  return (
    <footer className="bg-[var(--color-text)] text-white mt-24">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12 md:py-16">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-8 md:gap-12">
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="col-span-2 md:col-span-1"
          >
            <span className="text-xl font-bold text-[var(--color-primary)]">CARTLIFT AI</span>
            <p className="mt-3 text-white/70 text-sm leading-relaxed">
              Context-aware add-on recommendations for food delivery.
            </p>
          </motion.div>
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
          >
            <h4 className="font-semibold text-sm uppercase tracking-wider text-white/90">Company</h4>
            <ul className="mt-4 space-y-2">
              {links.company.map((l) => (
                <li key={l.label}>
                  <Link to={l.href} className="text-white/70 hover:text-white text-sm transition-colors">
                    {l.label}
                  </Link>
                </li>
              ))}
            </ul>
          </motion.div>
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
          >
            <h4 className="font-semibold text-sm uppercase tracking-wider text-white/90">Support</h4>
            <ul className="mt-4 space-y-2">
              {links.support.map((l) => (
                <li key={l.label}>
                  <Link to={l.href} className="text-white/70 hover:text-white text-sm transition-colors">
                    {l.label}
                  </Link>
                </li>
              ))}
            </ul>
          </motion.div>
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
          >
            <h4 className="font-semibold text-sm uppercase tracking-wider text-white/90">Legal</h4>
            <ul className="mt-4 space-y-2">
              {links.legal.map((l) => (
                <li key={l.label}>
                  <Link to={l.href} className="text-white/70 hover:text-white text-sm transition-colors">
                    {l.label}
                  </Link>
                </li>
              ))}
            </ul>
          </motion.div>
        </div>
        <div className="mt-12 pt-8 border-t border-white/20 flex flex-col sm:flex-row justify-between items-center gap-4">
          <p className="text-white/60 text-sm">© {new Date().getFullYear()} CARTLIFT AI. All rights reserved.</p>
          <p className="text-white/50 text-xs">CARTLIFT AI 2026</p>
        </div>
      </div>
    </footer>
  );
}
