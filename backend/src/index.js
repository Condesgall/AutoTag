import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import uploadRoutes from './routes/upload.routes.js';

dotenv.config();

const app = express();
app.use(cors());
app.use(express.json());

app.get('/', (req, res) => {
  res.send('Backend is running...');
});

<<<<<<< HEAD
=======
// Video uploading routes
app.use('/api', uploadRoutes);

>>>>>>> 3aef37314b0bc17a53c6c0a535401fa7728c972c
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
