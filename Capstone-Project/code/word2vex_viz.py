# The code for this visualization comes from this youtube video: https://www.youtube.com/watch?v=6PYeLFh-N1E
import gensim
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load from file
my_model = gensim.models.Word2Vec.load('../results/w2v_model.pkl')

# Get the vocabulary
vocab = my_model.wv.vocab

# Instantiate PCA object
pca = PCA(n_components=2)

# Fit and transform the data by subsetting the model with the vocabulary
my_pca = pca.fit_transform(my_model[vocab])

# Plot the data
plt.figure(figsize=(100,100))
plt.scatter(my_pca[:, 0], my_pca[:,1])

# Annotate the plot
for i, word in enumerate(vocab):
    plt.annotate(word, xy=(my_pca[i,0], my_pca[i, 1]))
    
plt.show()