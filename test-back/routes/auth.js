// routes/auth.js
const express = require('express');
const jwt = require('jsonwebtoken');
const User = require('../models/User');

const router = express.Router();

// Register Route
router.post('/register', async (req, res) => {
    const { username, password } = req.body;

    try {
        const newUser = new User({ username, password });
        await newUser.save();
        res.status(201).json({ message: 'User registered successfully' });
    } catch (error) {
        res.status(400).json({ error: 'Error registering user' });
    }
});

// Login Route
router.post('/login', async (req, res) => {
    const { username, password } = req.body;

    try {
        const user = await User.findOne({ username });
        if (!user || !(await user.comparePassword(password))) {
            return res.status(401).json({ error: 'Invalid credentials' });
        }

        // Create JWT
        const token = jwt.sign({ id: user._id }, 'secret_key', { expiresIn: '1h' });

        // Set cookie
        res.cookie('auth_token', token, {
            httpOnly: true,
            secure: true,
            sameSite: true,
            maxAge: 24 * 60 * 60 * 1000 // 1 day in milliseconds
        });
        res.status(200).json({ message: 'Login successful' });
    } catch (error) {
        res.status(500).json({ error: 'Server error' });
    }
});

// Protected Route Example
router.get('/protected', (req, res) => {
    const token = req.cookies.auth_token;
    if (!token) return res.sendStatus(403);

    jwt.verify(token, 'secret_key', (err, user) => {
        if (err) return res.sendStatus(403);
        res.json({ message: 'This is protected data', user });
    });
});

// Get User Info Route
router.get('/user-info', async (req, res) => {
    const token = req.cookies.auth_token; // Get the token from the cookie
    if (!token) return res.sendStatus(403); // Forbidden if no token

    jwt.verify(token, 'secret_key', async (err, user) => {
        if (err) return res.sendStatus(403); // Forbidden if token is invalid
        // Fetch user information from the database
        try {
            const userInfo = await User.findById(user.id).select('-password'); // Exclude password
            res.json(userInfo);
        } catch (error) {
            res.status(500).json({ error: 'Server error' });
        }
    });
});

// Logout route
router.post('/logout', (req, res) => {
    res.clearCookie('auth_token'); // Clear the cookie
    res.status(200).json({ message: 'Logged out successfully' });
});


// Token Validation Route
router.get('/check-token', async (req, res) => {
    const token = req.cookies.auth_token;
    if (!token) return res.json({ isLoggedIn: false });

    try {
        const decoded = jwt.verify(token, process.env.JWT_SECRET);
        
        // Additional check: Verify the user still exists
        const user = await User.findById(decoded.id);
        if (!user) {
            res.clearCookie('auth_token');
            return res.json({ isLoggedIn: false });
        }

        res.json({ isLoggedIn: true });
    } catch (err) {
        res.clearCookie('auth_token');
        res.json({ isLoggedIn: false });
    }
});

module.exports = router;
