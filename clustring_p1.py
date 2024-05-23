import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import style
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import sys
from scipy import stats
import io
style.use('ggplot')


class K_Means:
    def __init__(self, k=2, tol=0.001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
        self.labels = None
        self.final_centroids = None

    def fit(self, data):
        self.centroids = data[np.random.choice(data.shape[0], self.k, replace=False)]
        
        print("initial Centroids")
        print(self.centroids)
        for _ in range(self.max_iter):
            self.classifications = {i: [] for i in range(self.k)}
            self.labels = self._assign_labels(data)

            for i, featureset in enumerate(data):
                self.classifications[self.labels[i]].append(featureset)

            prev_centroids = np.copy(self.centroids)

            for classification in self.classifications:
                self.centroids[classification] = np.average(
                    self.classifications[classification], axis=0)

            print("Centroids in iteration", _ + 1)
            print(self.centroids)

            optimized = True

            for i in range(self.k):
                if np.sum((self.centroids[i] - prev_centroids[i]) / prev_centroids[i] * 100.0) > self.tol:
                    optimized = False

            if optimized:
                self.final_centroids = self.centroids
                break

        if self.final_centroids is not None:
            print("Final Centroids:")
            print(self.final_centroids)

    def _assign_labels(self, data):
        distances = np.sqrt(
            ((data - self.centroids[:, np.newaxis]) ** 2).sum(axis=2))
        return np.argmin(distances, axis=0)


def read_data(file_path, percentage):
    file_extension = file_path.split('.')[-1]
    if file_extension.lower() == 'csv':
        data = pd.read_csv(file_path)
    elif file_extension.lower() in ['xls', 'xlsx']:
        data = pd.read_excel(file_path)
    elif file_extension.lower() == 'txt':
        data = pd.read_csv(file_path, sep='\t')
    else:
        raise ValueError(
            "Unsupported file format. Please provide a CSV, Excel, or text file.")
    original_length = len(data)
    data = data.sample(frac=percentage, random_state=1)
    sampled_length = len(data)
    return data  , original_length, sampled_length


def analyze_data(file_path, percentage, k):
    output_buffer = io.StringIO()
    sys.stdout = output_buffer

    movie_data, original_length, sampled_length = read_data(file_path, percentage)

    print("Number of rows before sampling:", original_length)
    print("Number of rows after sampling:", sampled_length)
    movie_data.dropna(inplace=True)

    movie_data.drop_duplicates(
        subset=['Movie Name', 'IMDB Rating'], keep='first', inplace=True)

    Q1 = np.percentile(movie_data['IMDB Rating'], 25)
    Q3 = np.percentile(movie_data['IMDB Rating'], 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = movie_data[(movie_data['IMDB Rating'] < lower_bound) | (
        movie_data['IMDB Rating'] > upper_bound)]

    num_outliers_before = len(outliers)
    print("---------------------------------------")
    print("Number of outliers before clustering:", num_outliers_before)
    print("Outliers before clustering:")
    print(outliers[['Movie Name', 'IMDB Rating']])
    print("---------------------------------------")
    

    # Plot movie data before clustering
    # plt.figure(figsize=(10, 6))
    # plt.scatter(movie_data['IMDB Rating'], np.zeros_like(movie_data['IMDB Rating']), c='blue', s=50, alpha=0.5)
    # plt.scatter(outliers['IMDB Rating'], np.zeros_like(outliers['IMDB Rating']), c='red', marker='x', s=200, label='Outliers')
    # plt.xlabel('IMDB Rating')
    # plt.title('Movie Data Before Clustering')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    movie_data = movie_data[~movie_data.index.isin(outliers.index)]
    movie_data.reset_index(drop=True, inplace=True)

    X = movie_data[['IMDB Rating']].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = K_Means(k)
    kmeans.fit(X)

    for i in range(k):
        print("---------------------------------------")
        print(f"\nCluster {i+1}:")
        cluster_data = movie_data.iloc[kmeans.labels == i]
        num_points = len(cluster_data)
        print("Number of points in cluster:", num_points)
        print(cluster_data[['Movie Name', 'IMDB Rating']])
        print("---------------------------------------")

        Q1 = np.percentile(cluster_data['IMDB Rating'], 25)
        Q3 = np.percentile(cluster_data['IMDB Rating'], 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        cluster_outliers = cluster_data[(cluster_data['IMDB Rating'] < lower_bound) | (
            cluster_data['IMDB Rating'] > upper_bound)]
        print("---------------------------------------")
        print("Outliers in Cluster Number {", i+1, "}:")
        print(cluster_outliers[['Movie Name', 'IMDB Rating']])
        print("Number of outliers in Cluster Number", i+1, ":", len(cluster_outliers))
        print("---------------------------------------")

    # plt.figure(figsize=(10, 6))
    # colors = ['r', 'g', 'b', 'y', 'c', 'm']
    # for i in range(k):
    #     cluster_data = np.array(kmeans.classifications[i])
    #     plt.scatter(cluster_data[:, 0], np.zeros_like(cluster_data[:, 0]), c=colors[i], s=50, alpha=0.5)

    # for i in range(k):
    #     plt.scatter(kmeans.centroids[i][0], np.zeros_like(kmeans.centroids[i][0]), marker='x', color='k', s=200)

    # plt.scatter(outliers['IMDB Rating'], np.zeros_like(outliers['IMDB Rating']), c='red', marker='x', s=200, label='Outliers')
    # plt.xlabel('IMDB Rating')
    # plt.title('K-Means Clustering of Movie Data with Outliers')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    output_text = output_buffer.getvalue()

    sys.stdout = sys.__stdout__

    return output_text



def browse_file():
    file_path = filedialog.askopenfilename()
    file_entry.delete(0, tk.END)
    file_entry.insert(0, file_path)


def show_result():
    try:
        file_path = file_entry.get()
        percentage = int(percentage_entry.get()) / 100
        k = int(k_entry.get())
        output_text = analyze_data(file_path, percentage, k)
        console_output.delete('1.0', tk.END)
        console_output.insert(tk.END, output_text)
    except Exception as e:
        messagebox.showerror("Error", str(e))


root = tk.Tk()
root.title("Movie Data Analysis")

frame0 = tk.Frame(root)
frame0.pack()

file_label = tk.Label(frame0, text="File Path:")
file_label.grid(row=0, column=0, padx=5, pady=5)

file_entry = tk.Entry(frame0, width=50)
file_entry.grid(row=0, column=1, padx=5, pady=5)

file_button = tk.Button(frame0, text="Browse", command=browse_file)
file_button.grid(row=0, column=2, padx=5, pady=5)

percentage_label = tk.Label(frame0, text="Enter Percentage of Data (0-100) %:")
percentage_label.grid(row=1, column=0, padx=5, pady=5)

percentage_entry = tk.Entry(frame0)
percentage_entry.grid(row=1, column=1, padx=5, pady=5)

k_label = tk.Label(frame0, text="Enter Number of Clusters (k):")
k_label.grid(row=2, column=0, padx=5, pady=5)

k_entry = tk.Entry(frame0)
k_entry.grid(row=2, column=1, padx=5, pady=5)

analyze_button = tk.Button(frame0, text="Run", command=show_result)
analyze_button.grid(row=3, column=1, padx=5, pady=5)

frame1 = tk.Frame(root)
frame1.pack()

console_output = scrolledtext.ScrolledText(frame1, height=20, width=100)
console_output.pack()

root.mainloop()
