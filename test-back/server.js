// server.js
const express = require('express');
const mongoose = require('mongoose');
const cookieParser = require('cookie-parser');
const cors = require('cors');
const dotenv = require('dotenv');
const path = require('path');


dotenv.config(); // Load environment variables

const app = express();
const PORT = process.env.PORT || 3000;



// CORS configuration
const corsOptions = {
  origin: 'http://localhost:4200', // Allow requests from this origin
  credentials: true, // Allow credentials (cookies, authorization headers)
};

// Middleware
app.use(cors(corsOptions)); // Use CORS with options
//app.use(express.json()); // Parse JSON bodies
app.use(cookieParser()); // Parse cookies
app.use('/uploaded_images', express.static(path.join(__dirname, 'src', 'upload_folder')));
//app.use(express.urlencoded({ extended: true })); // For form-data (text fields)


app.use(express.json({ limit: '500mb' }));
app.use(express.urlencoded({ limit: '500mb', extended: true }));

// MongoDB Connection
mongoose.connect(process.env.MONGODB_URI)
  .then(() => console.log('MongoDB connected'))
  .catch(err => console.error('MongoDB connection error:', err));

// Importing routes
const authRoutes = require('./routes/auth');
const imageRoutes = require('./routes/imageCrud');

// Use auth routes
app.use('/api/auth', authRoutes);
app.use('/api/images', imageRoutes);

// Express route example
app.post('/set-preferences', (req, res) => {
    const { theme } = req.body;
    
    // Set first-party cookie for user preferences
    res.cookie('user_preference', JSON.stringify({ theme }), {
      maxAge: 86400000, // Expires in 1 day
      httpOnly: true,
      secure: true,
      sameSite: 'Strict'
    });
  
    // Set an analytics cookie (e.g., using a tracking library)
    res.cookie('_ga', 'GA_TRACKING_ID', {
      maxAge: 63072000000, // 2 years
      secure: true,
      sameSite: 'Strict'
    });
  
    res.json({ message: 'Preferences set and analytics cookie created.' });
  });

// Start Server
app.listen(PORT, () => {
  console.log(`Server is running on http://localhost:${PORT}`);
});
