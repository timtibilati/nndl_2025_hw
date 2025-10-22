# 🎨 Neural Style Transfer Web App (Flask + PyTorch)

This is a simple **Flask web application** that allows users to upload a photo and transform it into the style of one of **seven famous artists** — **Van Gogh, Picasso, Dali, Munch, Klimt, Chagall,** or **Mucha** — using **Neural Style Transfer (NST)** with PyTorch (work is based on topic [Neural Transfer Using PyTorch](https://docs.pytorch.org/tutorials/advanced/neural_style_tutorial.html)).

---

## 🧠 How It Works

1. The user uploads a **content image**.
2. Selects a **style** (an image of a painting by the chosen artist).
3. Flask calls the function `run_style_transfer()` from `neural_style.py`, which:

   * Loads the **VGG19** model from `torchvision.models`;
   * Extracts **content features** and **style features**;
   * Optimizes a new image to **preserve content structure** while adopting **style textures**.
4. The stylized result is saved in `/static/results/` and displayed to the user.

---

## 🗂 Project Structure

```
flask_style_transfer/
│
├── app.py                  # Main Flask application
├── neural_style.py         # Neural Style Transfer logic
├── requirements.txt
├── README.md
│
├── static/
│   ├── uploads/            # User uploaded images
│   ├── results/            # Stylized output images
│   └── models/
│       ├── van_gogh.jpg
│       ├── picasso.jpg
│       ├── dali.jpg
│       ├── munch.jpg
│       ├── klimt.jpg
│       ├── chagall.jpg
│       └── mucha.jpg
└── templates/
    ├── index.html          # Main upload page
    └── result.html         # Result page
```

---

## 🚀 Installation & Running

### 🧩 Option 1 — Local Setup

#### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/flask_style_transfer.git
cd flask_style_transfer
```

#### 2. Install dependencies

It is recommended to use a virtual environment:

```bash
python -m venv venv
source venv/bin/activate       # macOS / Linux
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

#### 3. Run the application

```bash
python app.py
```

After running, Flask will display:

```
Running on http://127.0.0.1:5000/
```

Open [http://127.0.0.1:5000/](http://127.0.0.1:5000/) in your browser.

---

### 🧩 Option 2 — GitHub Codespaces or Hugging Face Spaces

#### 1. Open a Codespace or Space based on this repository.

#### 2. Install dependencies

```bash
pip install -r requirements.txt
```

#### 3. Start Flask

```bash
python app.py --host=0.0.0.0 --port=7860
```

Flask will provide a public link to access the app.

---

## 🧰 Dependencies

`requirements.txt`:

```
flask
torch
torchvision
pillow
```

---

## 🖼 Available Styles

The style images are stored in `/static/models/`. You can replace them with your own images (preferably up to 512×512 px):

| Artist           | File           |
| ---------------- | -------------- |
| Vincent Van Gogh | `van_gogh.jpg` |
| Pablo Picasso    | `picasso.jpg`  |
| Salvador Dali    | `dali.jpg`     |
| Edvard Munch     | `munch.jpg`    |
| Gustav Klimt     | `klimt.jpg`    |
| Marc Chagall     | `chagall.jpg`  |
| Alphonse Mucha   | `mucha.jpg`    |

---

## 💡 Technical Details

* Uses the pre-trained **VGG19** convolutional network.
* Extracts **content features** from deeper layers and **style features** from multiple layers.
* Optimization with **L-BFGS** updates the pixels of the generated image to minimize:

```
loss = content_loss + style_loss * weight
```

where `style_loss` is based on **Gram matrices** (feature correlations).
After ~200 iterations, a stylized image is produced.

---

## 🖥 Interface

* Minimalistic but visually pleasant **Bootstrap 5** interface.
* Upload a photo, select a style, wait for processing, and view the stylized image.

---

## ⚙️ Settings & Optimization

Parameters in `neural_style.py`:

* `num_steps=200` — number of optimization iterations.
* `imsize=512` — image resolution.
* GPU can be used if available (`torch.device("cuda")`) for faster processing.

---

## 🧾 License

MIT License © 2025
Created for educational and demonstration purposes.
