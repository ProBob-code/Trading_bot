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

    // Arriving here from a page that needs a different account (?switch=1).
    // Without this the check below bounces an already-signed-in visitor home,
    // so someone logged in as an ordinary user could never reach this form to
    // sign in as the administrator — a dead end with no way out.
    const params = new URLSearchParams(window.location.search);
    const switching = params.has('switch');

    // `next` comes off the URL, so it is attacker-controllable. Only same-origin
    // absolute paths are honoured: anything scheme-relative ("//evil.com") or
    // absolute would turn this page into an open redirect that borrows the
    // site's credibility to land users somewhere else.
    const requestedNext = params.get('next') || '';
    const nextUrl = (/^\/[^\/\\]/.test(requestedNext)) ? requestedNext : '/godbot_home';

    if (switching) {
        showNotice('Sign in with an account that has access to that page.');
    } else {
        // Check if already logged in
        fetch('/api/auth/status')
            .then(res => res.json())
            .then(data => {
                if (data.authenticated) {
                    window.location.href = '/godbot_home';
                }
            });
    }

    /** A neutral hint, distinct from the red error styling. */
    function showNotice(msg) {
        if (!successMsg) return;
        successMsg.textContent = msg;
        successMsg.classList.remove('hidden');
        errorMsg.classList.add('hidden');
    }

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
                
                if (data._demo_otp) {
                    successMsg.textContent = `Code sent! [DEMO MODE] Your OTP is: ${data._demo_otp}`;
                    console.log(`[DEMO MODE] Received OTP: ${data._demo_otp}`);
                    
                    // Auto-fill the OTP input boxes
                    const otpChars = data._demo_otp.split('');
                    otpInputs.forEach((input, idx) => {
                        if (otpChars[idx]) input.value = otpChars[idx];
                    });
                } else {
                    successMsg.textContent = 'Code sent! Check your system logs.';
                }
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
                setTimeout(() => window.location.href = '/godbot_home', 1500);
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
                // Use replace to prevent back button from returning to login
                window.location.replace(nextUrl);
            } else {
                showError(data.error || 'Login failed');
            }
        } catch (err) {
            console.error(err);
        }
    });

    function showError(msg) {
        errorMsg.textContent = msg;
        errorMsg.classList.remove('hidden');
        successMsg.classList.add('hidden');
    }
});
