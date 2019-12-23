CS583: Deep Learning
============


> Instructor: Shusen Wang


Description
---------

**Meeting Time:**

- Thursday, 6:30-9:00 PM, TBD


**Office Hours:**

- Thursday, 3:00 - 5:00 PM, Gateway South 354



**Contact the Instructor:**

- For questions regarding grading, talk to the instructor during office hours or send him emails.

- For any other questions, come during the office hours; the instructor will NOT reply such emails.


**Prerequisite:**

- Elementary linear algebra, e.g., matrix multiplication, eigenvalue decomposition, and matrix norms.

- Elementary calculus, e.g., convex function, differentiation of scalar functions, first derivative, and second derivative.

- Python programming (especially the Numpy library) and Jupyter Notebook.


**Goal:** This is a practical course; the students will be able to use DL methods for solving real-world ML, CV, and NLP problems. The students will also learn math and theories for understanding ML and DL.



Schedule
---------


- Preparations

	* Install the software packages by following [[this](https://github.com/wangshusen/CS583-2019F/blob/master/homework/Prepare/HM.pdf)]
	
	* Study elementary matrix algebra by following [[this book](http://vmls-book.stanford.edu/vmls.pdf)]
	
	* Finish the [[sample questions](https://github.com/wangshusen/CS583-2019F/blob/master/homework/Quiz1-Sample/Q1.pdf)] before Quiz 1.


- Jan 23, Lecture 1

    * Fundamental ML problems
    
    * Regression
    
    
- Jan 30, Lecture 2

    * Read these before coming: 
    [[Matrix Calculus](https://github.com/wangshusen/CS583A-2019Spring/blob/master/reading/MatrixCalculus.pdf)]
    [[Logistic Regression](https://github.com/wangshusen/DeepLearning/blob/master/LectureNotes/Logistic/paper/logistic.pdf)]
    
    * Regression (Cont.)
    
    * Classification: logistic regression and SVM.
    
    
    
- Feb 6, Lecture 3
    
    * Classification: softmax classifier and KNN.



- Feb 6, **Quiz 1** after the lecture (around 5\% of the total)

	* Coverage: vectors norms ($\ell_2$-norm, $\ell_1$-norm, $\ell_p$-norm, $\ell_\infty$-norm), vector inner product, matrix multiplication, matrix trace, matrix Frobenius norm, scalar function differential, convex function, use Numpy to construct vectors and matrices.
	
	* Policy: Printed material is allowed. No electronic device (except for electronic calculator). 
  


- Feb 13, Lecture 4

    * Read Sections 1 to 3 before coming: [[SVD and PCA](https://github.com/wangshusen/DeepLearning/blob/master/LectureNotes/SVD/svd.pdf)]

    * Read Sections 1 to 4 before coming: [[neural networks and backpropagation](https://github.com/wangshusen/DeepLearning/blob/master/LectureNotes/BP/bp.pdf)]

    * Regularization.
    
    * Singular value decomposition (SVD).
    
    * Scientific computing libraries.
    
    * Neural networks.


- Feb 20, Lecture 5
    
    * Keras.
    
    * Convolutional neural networks (CNNs).


- Feb 27, Lecture 6

    * Finish reading the note before coming: [[neural networks and backpropagation](https://github.com/wangshusen/DeepLearning/blob/master/LectureNotes/BP/bp.pdf)]
    
    * CNNs (Cont.)


- Mar 5, Lecture 7
    
    * CNNs (Cont.)
    

- Mar 12, Lecture 8

    * Recurrent neural networks (RNNs).
    





Assignments and Bonus Scores
---------

- Homework 1: Linear Algebra Basics

	* Available only on Canvas (auto-graded.)
	
	* Submit to Canvas before Feb 6.
	
 
- Homework 2: Implement Numerical Optimization Algorithms (Voluntary)

	* Available at the course's repo [[click here](https://github.com/wangshusen/CS583-2020S/tree/master/homework)].
	* You will need the knowledge in the lecture note: [[Logistic Regression](https://github.com/wangshusen/DeepLearning/blob/master/LectureNotes/Logistic/paper/logistic.pdf)]
	
	* Submit to Canvas before Feb 23.

 	
- Homework 3: Machine Learning Basics

	* Available only on Canvas (auto-graded.)

	* Submit to Canvas before Mar 8.
	
 
- Homework 4: Implement a Convolutional Neural Network

	* Available at the course's repo [[click here](https://github.com/wangshusen/CS583-2020S/tree/master/homework)].
	
	* Submit to Canvas before Mar 22.
	
 
- Homework 5: Implement a Recurrent Neural Network

	* Available at the course's repo [[click here](https://github.com/wangshusen/CS583-2020S/tree/master/homework)].
	
	* Submit to Canvas before Apr 12.
	
	* You may get up to 2 bonus scores by doing extra work. 
	
	
- Bonus 1: Implement Parallel Algorithms (Voluntary)

	* Available at Canvas.
	
	* You will need the knowledge in the lecture note: [[Parallel Computing](https://github.com/wangshusen/DeepLearning/blob/master/LectureNotes/Parallel/Parallel.pdf)]
	
	* You can choose to implement Federated Averaging or/and Decentralized Optimization. You may get up to 2 bonus points for each. 
	
	* Submit to Canvas before Apr 5 (firm deadline).

 
- Bonus 2: Implement an Autoencoder Network (Voluntary)

	* Available at the course's repo [[click here](https://github.com/wangshusen/CS583-2020S/tree/master/homework)].
	
	* You may get up to 1 bonus point. 
	
	* Submit to Canvas before May 1 (firm deadline).	
	


 
Syllabus and Slides
---------

1. **Machine learning basics.**
This part briefly introduces the fundamental ML problems-- regression, classification, dimensionality reduction, and clustering-- and the traditional ML models and numerical algorithms for solving the problems.

    * ML basics. 
    [[slides-1](https://github.com/wangshusen/DeepLearning/blob/master/Slides/1_ML_Basics.pdf)]
    [[slides-2](https://github.com/wangshusen/DeepLearning/blob/master/Slides/1_Models.pdf)]

    
    * Regression. 
    [[slides-1](https://github.com/wangshusen/DeepLearning/blob/master/Slides/2_Regression_1.pdf)] 
    [[slides-2](https://github.com/wangshusen/DeepLearning/blob/master/Slides/2_Regression_2.pdf)]
    
    
    * Classification. 
    
        - Logistic regression: 
        [[slides](https://github.com/wangshusen/DeepLearning/blob/master/Slides/3_Classification_1.pdf)] 
        [[lecture note](https://github.com/wangshusen/DeepLearning/blob/master/LectureNotes/Logistic/paper/logistic.pdf)]
    
        - SVM: [[slides](https://github.com/wangshusen/DeepLearning/blob/master/Slides/3_Classification_2.pdf)] 
    
        - Softmax classifier: [[slides](https://github.com/wangshusen/DeepLearning/blob/master/Slides/3_Classification_3.pdf)] 
    
        - KNN classifier: [[slides](https://github.com/wangshusen/DeepLearning/blob/master/Slides/3_Classification_4.pdf)]
    
    * Regularizations. 
    [[slides-1](https://github.com/wangshusen/DeepLearning/blob/master/Slides/3_Optimization.pdf)]
    [[slides-2](https://github.com/wangshusen/DeepLearning/blob/master/Slides/3_Regularizations.pdf)]
    
    * Clustering. 
    [[slides](https://github.com/wangshusen/DeepLearning/blob/master/Slides/4_Clustering.pdf)] 
    
    * Dimensionality reduction. 
    [[slides-1](https://github.com/wangshusen/DeepLearning/blob/master/Slides/5_DR_1.pdf)] 
    [[slides-2](https://github.com/wangshusen/DeepLearning/blob/master/Slides/5_DR_2.pdf)] 
    [[lecture note](https://github.com/wangshusen/DeepLearning/blob/master/LectureNotes/SVD/svd.pdf)]
    
    * Scientific computing libraries.
    [[slides](https://github.com/wangshusen/DeepLearning/blob/master/Slides/5_DR_3.pdf)]
    
    
2. **Neural network basics.**
This part covers the multilayer perceptron, backpropagation, and deep learning libraries, with focus on Keras.

    * Multilayer perceptron and backpropagation. 
    [[slides](https://github.com/wangshusen/DeepLearning/blob/master/Slides/6_NeuralNet_1.pdf)]
    [[lecture note](https://github.com/wangshusen/DeepLearning/blob/master/LectureNotes/BP/bp.pdf)]
    
    * Keras. 
    [[slides](https://github.com/wangshusen/DeepLearning/blob/master/Slides/6_NeuralNet_2.pdf)]
    
    * Further reading:
    
        - [[activation functions](https://adl1995.github.io/an-overview-of-activation-functions-used-in-neural-networks.html)]
        
        - [[parameter initialization](https://towardsdatascience.com/weight-initialization-in-neural-networks-a-journey-from-the-basics-to-kaiming-954fb9b47c79)]
    
        - [[optimization algorithms](http://ruder.io/optimizing-gradient-descent/)]
    
    
3. **Convolutional neural networks (CNNs).**
This part is focused on CNNs and its application to computer vision problems.

    * CNN basics.
    [[slides](https://github.com/wangshusen/DeepLearning/blob/master/Slides/7_CNN_1.pdf)]
    
    * Tricks for improving test accuracy.
    [[slides](https://github.com/wangshusen/DeepLearning/blob/master/Slides/7_CNN_2.pdf)]
    
    * Feature scaling and batch normalization.
    [[slides](https://github.com/wangshusen/DeepLearning/blob/master/Slides/7_CNN_3.pdf)]
    
    * Advanced topics on CNNs. 
    [[slides](https://github.com/wangshusen/DeepLearning/blob/master/Slides/7_CNN_4.pdf)]
    
    * Popular CNN architectures.
    [[slides](https://github.com/wangshusen/DeepLearning/blob/master/Slides/7_CNN_5.pdf)]
    
    * Face recognition.
    [[slides](https://github.com/wangshusen/DeepLearning/blob/master/Slides/7_CNN_6.pdf)]
    
    * Further reading: 
    
        - [style transfer (Section 8.1, Chollet's book)]
        
        - [visualize CNN (Section 5.4, Chollet's book)]



4. **Recurrent neural networks (RNNs).**
This part introduces RNNs and its applications in natural language processing (NLP).

    * Text processing.
    [[slides](https://github.com/wangshusen/DeepLearning/blob/master/Slides/9_RNN_1.pdf)] 
       
    * RNN basics and LSTM.
    [[slides](https://github.com/wangshusen/DeepLearning/blob/master/Slides/9_RNN_2.pdf)]
    [[reference](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)]
   
    * Text generation.
    [[slides](https://github.com/wangshusen/DeepLearning/blob/master/Slides/9_RNN_3.pdf)]
    
    * Machine translation. 
    [[slides](https://github.com/wangshusen/DeepLearning/blob/master/Slides/9_RNN_4.pdf)]
    
    * Image caption generation. 
    [[slides](https://github.com/wangshusen/DeepLearning/blob/master/Slides/9_RNN_5.pdf)]
    [[reference](https://machinelearningmastery.com/develop-a-deep-learning-caption-generation-model-in-python/)]
    
    * Attention. 
    [[slides](https://github.com/wangshusen/DeepLearning/blob/master/Slides/9_RNN_6.pdf)]
    [[reference-1](https://distill.pub/2016/augmented-rnns/)]
    [[reference-2](https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html)]
    
    
5. **Language Models beyond RNNs.**

    * Transformer model: beyond RNNs. 
    [[slides](https://github.com/wangshusen/DeepLearning/blob/master/Slides/9_RNN_7.pdf)]
    [[reference](https://arxiv.org/pdf/1706.03762.pdf)]
    
    * Pre-train Transformer using BERT. [[slides](https://github.com/wangshusen/DeepLearning/blob/master/Slides/9_RNN_8.pdf)]
    [[reference](https://arxiv.org/pdf/1810.04805.pdf)]


6. **Autoencoders.**
This part introduces autoencoders for dimensionality reduction and image generation.

    * Autoencoder for dimensionality reduction.
    [[slides](https://github.com/wangshusen/DeepLearning/blob/master/Slides/8_AE_1.pdf)]
    
    * Variational Autoencoders (VAEs) for image generation. 
    [[slides](https://github.com/wangshusen/DeepLearning/blob/master/Slides/8_AE_2.pdf)]

    
7. **Generative Adversarial Networks (GANs).** 

    * DC-GAN [[slides](https://github.com/wangshusen/DeepLearning/blob/master/Slides/12_GAN.pdf)]



8. **Recommender system.** 
This part is focused on the collaborative filtering approach to recommendation based on the user-item rating data.
This part covers matrix completion methods and neural network approaches. 

    * Collaborative filtering. 
    [[slides](https://github.com/wangshusen/DeepLearning/blob/master/Slides/10_Recommender.pdf)]

    
9. **Deep Reinforcement Learning.** 

    * Reinforcement learning [[slides](https://github.com/wangshusen/DeepLearning/blob/master/Slides/13_RL_1.pdf)] [[lecture note](https://github.com/wangshusen/DeepLearning/blob/master/LectureNotes/DRL/DRL.pdf)]

    * Value-based learning [[slides](https://github.com/wangshusen/DeepLearning/blob/master/Slides/13_RL_2.pdf)]

    * Policy-based learning [[slides](https://github.com/wangshusen/DeepLearning/blob/master/Slides/13_RL_3.pdf)]

    * Actor-critic methods [[slides](https://github.com/wangshusen/DeepLearning/blob/master/Slides/13_RL_4.pdf)]

    * AlphaGo [[slides](https://github.com/wangshusen/DeepLearning/blob/master/Slides/13_RL_5.pdf)]


10. **Parallel Computing.** 

	* Basics and MapReduce. [[slides](https://github.com/wangshusen/DeepLearning/blob/master/Slides/14_Parallel_1.pdf)] [[lecture note](https://github.com/wangshusen/DeepLearning/blob/master/LectureNotes/Parallel/Parallel.pdf)] [[Video (in Chinese)](https://youtu.be/gVcnOe6_c6Q)]
	
	* Parameter Server and Decentralized Network. [[slides](https://github.com/wangshusen/DeepLearning/blob/master/Slides/14_Parallel_2.pdf)] [[Video (in Chinese)](https://youtu.be/Aga2Lxp3G7M)]
	
	* Federated Learning. [[slides](https://github.com/wangshusen/DeepLearning/blob/master/Slides/14_Parallel_3.pdf)] [[Video (in Chinese)](https://youtu.be/STxtRucv_zo)]


11. **Adversarial Robustness.**
This part introduces how to attack neural networks using adversarial examples and how to defend from the attack.

	* Data evasion attack and defense.
    [[slides](https://github.com/wangshusen/DeepLearning/blob/master/Slides/11_Adversarial.pdf)]
    [[lecture note](https://github.com/wangshusen/DeepLearning/blob/master/LectureNotes/Adversarial/DataAttacks.pdf)]
        
    * Further reading:
    [[Adversarial Robustness - Theory and Practice](https://adversarial-ml-tutorial.org/)]
    



Textbooks
---------

**Required** (Please notice the difference between "required" and "recommended"):

- Francois Chollet. Deep learning with Python. Manning Publications Co., 2017. (Available online.)

**Highly Recommended**:

- S. Boyd and L. Vandenberghe. Introduction to Applied Linear Algebra. Cambridge University Press, 2018. (Available online.)

**Recommended**:

- Y. Nesterov. Introductory Lectures on Convex Optimization Book. Springer, 2013. (Available online.)

- D. S. Watkins. Fundamentals of Matrix Computations. John Wiley & Sons, 2004.

- I. Goodfellow, Y. Bengio, A. Courville, Y. Bengio. Deep learning. MIT press, 2016. (Available online.)
    
- M. Mohri, A. Rostamizadeh, and A. Talwalkar. Foundations of machine learning. MIT press, 2012.
    
- J. Friedman, T. Hastie, and R. Tibshirani. The elements of statistical learning. Springer series in statistics, 2001. (Available online.)



Grading Policy
---------

**Weights**:

- Homework 50\%

- Quizzes 25\% (students' average score is likely around 20)

- Final 25\% (students' average score is likely around 20)

- Bonus (5\% from bonus assignments; 3\% from HM4.)


**Expected grade on record**:

- An average student is expected to lose at least 5+5=10 points. 

- If an average student does not collect any bonus score, his grade on record is expected to be "B+".
An average student needs at least 3 bonus scores to get "A".

- According to Stevens's policy, a score lower than 73.0 will be fail.


**Late penalty**:

- Late submissions of assignments or project document for whatever reason will be punished. 2\% of the score of an assignment/project will be deducted per day. For example, if an assignment is submitted 15 days and 1 minute later than the deadline (counted as 16 days) and it gets a grade of 95\%, then the score after the deduction will be: 95\% - 2*16\% = 63\%.

- All the deadlines for bonus are firm. Late submission will not receive bonus score.

- May 1 is the firm deadline for all the homework and the course project. Submissions later than the firm deadline will not be graded.


