const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const Image = require('../models/Image'); // Assuming this is your Mongoose model

const router = express.Router();

// Define valid categories
const validCategories = [
  "Alabastron", "Amphora", "Amphoriskos", "Aryballos", "Askos", "Bowl", "Cup", "Dinos",
  "Epichysis", "Exaleiptron", "Skyphos", "Hydria", "Kalathos", "Kantharos", "Kernos", 
  "Krater", "Kyathos", "Kylix", "Lagynos", "Lebes", "Lekane", "Lekythos", "Loutrophoros",
  "Lydion", "Mastos", "Mug", "Nestoris", "Oinochoe", "Pelike", "Pithos", "Plemochoe",
  "Psykter", "Pyxis", "Skyphos", "Other", "Modern-Bottle", "Modern-Vase", "Modern-Glass",
  "Modern-Bowl", "Modern-Cup", "Modern-Mug", "Modern-Urn", "Modern-Pot", "Pithoeidi",
  "Native American - Jar", "Native American - Effigy", "Native American - Bowl",
  "Native American - Bottle", "Picher Shaped", "Abstract"
];

// Function to create category folders dynamically
const createCategoryFolder = (category, subfolder) => {
  const folderPath = path.join(__dirname, `../src/upload_folder/${category}/${subfolder}`);
  fs.mkdirSync(folderPath, { recursive: true });
  return folderPath;
};

// Multer storage configuration
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    const { category } = req.body;

    if (!validCategories.includes(category)) {
      return cb(new Error(`Invalid category: ${category}`));
    }

    const subfolder = file.fieldname === 'previews' ? 'previews' : 'objects';
    const uploadDir = createCategoryFolder(category, subfolder);
    cb(null, uploadDir);
  },
  filename: (req, file, cb) => {
    const uniqueFilename = `${Date.now()}-${file.originalname}`;
    cb(null, uniqueFilename);
  }
});

// Multer file filter
const fileFilter = (req, file, cb) => {
  const { category } = req.body;

  if (!validCategories.includes(category)) {
    return cb(new Error(`Invalid or missing category. Valid categories: ${validCategories.join(', ')}`), false);
  }

  cb(null, true);
};

// Multer upload middleware for multiple files
const upload = multer({ 
  storage, 
  limits: { 
    fileSize: 50 * 1024 * 1024, // 50MB per file
    files: 500 // Allow up to 500 files per request
  },
  fileFilter
}).fields([
  { name: 'previews', maxCount: 500 }, 
  { name: 'objects', maxCount: 500 }
]);

// POST: Upload images & 3D objects
router.post('/upload', async (req, res) => {
  upload(req, res, async (err) => {
    if (err) {
      return res.status(400).json({ error: err.message });
    }

    try {
      const { category } = req.body;
      const previewFiles = req.files.previews || [];
      const objectFiles = req.files.objects || [];

      if (previewFiles.length !== objectFiles.length) {
        return res.status(400).json({ error: 'Number of previews and objects must be the same.' });
      }

      const uploadedFiles = await Promise.all(
        previewFiles.map(async (preview, index) => {
          const newImage = await Image.create({
            name: preview.filename, // Use filename instead of originalname to avoid duplicate issues
            category,
            previewPath: preview.path,
            objectPath: objectFiles[index]?.path || '',
          });
          return newImage;
        })
      );

      res.status(201).json({
        message: 'Files uploaded successfully!',
        files: uploadedFiles
      });

    } catch (error) {
      console.error('Error processing files:', error);
      res.status(500).json({ error: 'Server error during upload' });
    }
  });
});


// GET: Fetch all images & objects
router.get('/all', async (req, res) => {
  try {
    const images = await Image.find();
    res.status(200).json(images);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Error fetching images' });
  }
});

// DELETE: Delete an object by ID
router.delete('/delete/:id', async (req, res) => {
  try {
    const { id } = req.params;
    const image = await Image.findById(id);
    
    if (!image) {
      return res.status(404).json({ error: 'Object not found' });
    }

    const previewPath = path.join(__dirname, '../', image.previewPath);
    const objectPath = path.join(__dirname, '../', image.objectPath);

    if (fs.existsSync(previewPath)) fs.unlinkSync(previewPath);
    if (fs.existsSync(objectPath)) fs.unlinkSync(objectPath);

    await Image.findByIdAndDelete(id);

    res.status(200).json({ message: 'Object deleted successfully!' });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Error deleting object' });
  }
});

// GET: Fetch images & objects by category
router.get('/category/:category', async (req, res) => {
  try {
    const { category } = req.params;

    if (!validCategories.includes(category)) {
      return res.status(400).json({
        error: `Invalid category. Valid categories: ${validCategories.join(', ')}`,
      });
    }

    const images = await Image.find({ category });

    if (!images.length) {
      return res.status(404).json({ error: 'No objects found in this category.' });
    }

    res.status(200).json(images);
  } catch (err) {
    console.error('Error fetching objects by category:', err);
    res.status(500).json({ error: 'Error fetching objects' });
  }
});

module.exports = router;
