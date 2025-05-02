# IrisKNNClassifier
## Elevale AI & ML Internship - Task 6

### My Journey with KNN and Iris
Hey there! I’m super excited to share my work on Task 6 for Elevale’s AI & ML Internship. When I got this task, I was both pumped and a bit intimidated—KNN sounded cool, but I had to figure it out from scratch. I spent hours digging into blogs, YouTube tutorials, and Scikit-learn’s docs to wrap my head around it. This project is the result of late-night coding sessions, a few “oops” moments, and some real victories. I hope it shows how much I learned and how hard I worked to make it my own!

### What Was the Task?
Task 6 was about building a KNN classifier using Scikit-learn, normalizing features, testing different K values, checking accuracy and confusion matrices, and plotting decision boundaries. The Iris dataset was suggested, and I went with it because it’s a classic (and honestly, those flower names are pretty cool). My goal was to not just get it done but to add something unique that screamed “I thought about this!”

### The Iris Dataset
- **What’s It About?** The Iris dataset has 150 samples of three flower types: Iris-setosa, Iris-versicolor, and Iris-virginica. Each sample has four features: sepal length, sepal width, petal length, and petal width (all in cm).
- **File**: `Iris.csv` (I also had a SQL version, but I stuck with CSV because it was way easier to load with Pandas).
- **Why Iris?** It’s perfect for classification, and I could see why it’s a go-to for ML beginners like me. Plus, the data was clean, so I could focus on the algorithm.

### How I Tackled It
Here’s the story of how I built this project:
1. **Learning KNN**: I started by reading about KNN—how it finds the closest neighbors to classify points. I was mind-blown by how simple yet powerful it is! But I learned the hard way that distances matter, so normalizing features is a must.
2. **Loading the Data**: I used Pandas to read `Iris.csv`. I turned the species names into numbers (0, 1, 2) since KNN needs that. I also peeked at the data with `.head()` to make sure I wasn’t missing anything.
3. **Picking Features**: I noticed petal length and width separated the classes better (after some scatter plots I made to check). So, I used those for my 2D decision boundary plots but also tested all four features to compare.
4. **Normalizing**: I used `StandardScaler` to make all features play fair in the distance game. Without this, my early tests were all over the place!
5. **KNN Experiments**: I tried K=3, 5, 7, 9 because those seemed like reasonable numbers. I used Euclidean distance (the standard one) and got curious about Minkowski after reading about it. I set p=1.3 for Minkowski to try something different—it was my little experiment!
6. **Evaluating**: I calculated accuracy and made confusion matrices. I turned them into heatmaps because they’re easier to read than plain numbers.
7. **Visualizing**: Plotting the decision boundary was the toughest part. I struggled with mesh grids (so many errors!), but after some trial and error, I got it to show how KNN splits the classes. I also added a box plot to see why petal length matters so much.

### What’s in This Folder
Here’s everything I created:
- `knn_iris.py`: My Python script where all the magic happens.
- `Iris.csv`: The dataset I used.
- `cm_k3_euclidean.png`, `cm_k5_euclidean.png`, etc.: Heatmap confusion matrices for K=3, 5, 7, 9 with Euclidean and Minkowski metrics (8 files total).
- `decision_boundary.png`: A plot showing how KNN separates the three Iris classes using petal features (K=5, Euclidean).
- `accuracy_vs_k.png`: A line plot of accuracy for different K values.
- `petal_length_boxplot.png`: A box plot I made to explore petal length’s role in classification.

### My Results
After running my script, here’s what I found:
- **Euclidean Distance**:
  - K=3: 1.0000
  - K=5: 1.0000 (perfect score!)
  - K=7: 1.0000
  - K=9: 1.0000
- **Minkowski Distance (p=1.3)**:
  - K=3: 1.0000
  - K=5: 1.0000
  - K=7: 1.0000
  - K=9: 1.0000
- **All Features (K=5, Euclidean)**: 1.0000
- **What I Noticed**: I was amazed—everything hit 100% accuracy! Whether I used petal features, all features, Euclidean, or Minkowski, KNN nailed it every time. I think it’s because the Iris dataset is so well-structured, and the petal features are super good at separating the classes. It made me realize how powerful KNN can be when the data is clean!

### The Visuals
- **Confusion Matrices**: These heatmaps show where KNN got it right or wrong. Since I got 100% accuracy, they’re all perfect—no errors to see here!
- **Decision Boundary**: This plot is my favorite. It shows three colored zones for the Iris classes, with training and test points scattered. It really brings KNN to life.
- **Accuracy Plot**: This shows how accuracy changes with K. It’s a flat line at 100%, which was surprising but cool to see.
- **Box Plot**: I added this to see how petal length varies across species. Setosa’s petals are way shorter, which explains why petal features work so well!

### How to Run My Code
Want to try it? Here’s how:
1. Make sure you have Python and these libraries: `pip install numpy pandas scikit-learn matplotlib seaborn`
2. Put `Iris.csv` in the same folder as `knn_iris.py`.
3. Run it: `python knn_iris.py`
4. Check out the plots that pop up in the folder.

### What I Learned
This task was a rollercoaster, but I loved it! Here’s what stuck with me:
- KNN is like finding your closest friends to make a decision—it’s so intuitive.
- Normalizing features is non-negotiable. I tried skipping it once, and my accuracies were a mess.
- Feature selection is huge. Petal features were my MVPs after I saw those box plots.
- Plotting is tricky but worth it. Getting the decision boundary right felt like winning a prize.
- Debugging is part of the game. I spent an hour fixing a plot size issue, but it taught me to double-check everything.

### My Challenges and Wins
I hit a wall with the decision boundary plot—those mesh grids were confusing! I watched a YouTube video that explained `np.meshgrid`, and it finally clicked. My biggest win was the box plot idea. I wasn’t sure if it’d help, but when I saw how clearly it showed petal length differences, I was stoked. It felt like I was uncovering a secret about the data!

### A Little Extra
I had a SQL version of the dataset (`data.sql`), but I chose `Iris.csv` to keep things simple. I did try loading the SQL file in MySQL to see what it was about, and it was cool to learn, but CSV was faster for this task. I’m glad I explored both, though—it made me feel like a real data scientist!

### About Me
Hi, I’m [Your Full Name]! I’m a budding data enthusiast with a huge curiosity for AI and machine learning. I’ve been diving into Python and ML concepts for a while now, and this internship task was the perfect chance to put my skills to the test. I love exploring datasets, tinkering with algorithms, and turning data into stories—working on this KNN project felt like solving a fun puzzle! I’m eager to learn from Elevale’s mentors, grow my ML skills, and contribute to real-world AI solutions. When I’m not coding, you’ll probably find me reading up on new tech trends or sketching out ideas for my next project.

Thanks for checking out my IrisKNNClassifier project! This task pushed me to think, code, and learn in ways I didn’t expect. I’m so grateful for the chance to grow with Elevale, and I can’t wait to tackle more ML challenges!