/** @type {import('tailwindcss').Config} */
export default {
    content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
    darkMode: 'class',
    theme: {
        extend: {
            colors: {
                dark: {
                    'primary': '#1a1a1a',
                    'secondary': '#242424',
                    'text': 'rgba(255, 255, 255, 0.87)',
                    'accent': '#646cff',
                    'accent-hover': '#535bf2',
                },
                light: {
                    'primary': '#ffffff',
                    'secondary': '#f9f9f9',
                    'text': '#213547',
                    'accent': '#646cff',
                    'accent-hover': '#747bff',
                },
            },
        },
    },
    plugins: [],
}

