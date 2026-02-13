document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const loginForm = document.getElementById('loginForm');
    const signupForm = document.getElementById('signupForm');
    const otpForm = document.getElementById('otpForm');
    const formTitle = document.getElementById('formTitle');
    const errorMsg = document.getElementById('errorMsg');
    const successMsg = document.getElementById('successMsg');
    const otpInputs = document.querySelectorAll('.otp-input');
    const displayMobile = document.getElementById('displayMobile');

    let currentMobile = '';

    // Check if already logged in
    fetch('/api/auth/status')
        .then(res => res.json())
        .then(data => {
            if (data.authenticated && data.user.is_verified) {
                window.location.href = 'index.html';
            }
        });

    // Handle Form Switching
    window.toggleAuth = (type) => {
        errorMsg.classList.add('hidden');
        successMsg.classList.add('hidden');
        if (type === 'signup') {
            loginForm.classList.add('hidden');
            signupForm.classList.remove('hidden');
            otpForm.classList.add('hidden');
            formTitle.textContent = 'Account Verification';
        } else {
            loginForm.classList.remove('hidden');
            signupForm.classList.add('hidden');
            otpForm.classList.add('hidden');
            formTitle.textContent = 'Welcome Back';
        }
    };

    // Handle Signup (Mobile OTP)
    signupForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const mobile = document.getElementById('signupMobile').value;
        currentMobile = mobile;

        try {
            const res = await fetch('/api/auth/register', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ mobile })
            });
            const data = await res.json();

            if (data.success) {
                displayMobile.textContent = mobile;
                signupForm.classList.add('hidden');
                otpForm.classList.remove('hidden');
                formTitle.textContent = 'Verify OTP';
                successMsg.textContent = 'Code sent! Check your system logs (Simulation).';
                successMsg.classList.remove('hidden');
            } else {
                showError(data.error);
            }
        } catch (err) {
            showError('Server connection error');
        }
    });

    // Handle OTP Inputs focus
    otpInputs.forEach((input, index) => {
        input.addEventListener('input', (e) => {
            if (e.target.value && index < otpInputs.length - 1) {
                otpInputs[index + 1].focus();
            }
        });
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Backspace' && !e.target.value && index > 0) {
                otpInputs[index - 1].focus();
            }
        });
    });

    // Handle OTP Verification
    otpForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const otp = Array.from(otpInputs).map(i => i.value).join('');
        const username = document.getElementById('finalUsername').value;
        const password = document.getElementById('finalPassword').value;

        try {
            const res = await fetch('/api/auth/verify', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ mobile: currentMobile, otp, username, password })
            });
            const data = await res.json();

            if (data.success) {
                successMsg.textContent = 'Verification Successful! Redirecting...';
                successMsg.classList.remove('hidden');
                setTimeout(() => window.location.href = 'index.html', 1500);
            } else {
                showError(data.error);
            }
        } catch (err) {
            showError('Verification failed');
        }
    });

    // Handle Login
    loginForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const username = document.getElementById('loginUsername').value;
        const password = document.getElementById('loginPassword').value;

        try {
            const res = await fetch('/api/auth/login', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ username, password })
            });
            const data = await res.json();

            if (data.success) {
                window.location.href = 'index.html';
            } else {
                showError(data.error);
            }
        } catch (err) {
            showError('Login failed');
        }
    });

    function showError(msg) {
        errorMsg.textContent = msg;
        errorMsg.classList.remove('hidden');
        successMsg.classList.add('hidden');
    }
});
