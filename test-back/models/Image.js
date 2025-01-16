const mongoose = require('mongoose');

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

const imageSchema = new mongoose.Schema({
  name: { type: String, required: true, unique: true },
  category: { 
    type: String, 
    required: true,
    enum: validCategories
  },
  previewPath: { type: String, required: true },  // Store preview image path
  objectPath: { type: String, required: true },   // Store 3D object path
});

module.exports = mongoose.model('Image', imageSchema);
