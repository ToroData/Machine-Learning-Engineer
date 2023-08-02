# Operationalizing an AWS ML Project


<div align="center">
    <img src="https://thedatascientist.digital/img/logo.png" alt="Logo" width="25%">
</div>




In this project, you'll start with a completed ML project. The completed project contains code that trains and deploys an image classification model on AWS Sagemaker. Your goal in this project will be to use several important tools and features of AWS to adjust, improve, configure, and prepare the model you started with for production-grade deployment. Taking raw ML code and preparing it for production deployment is a common task for ML engineers.

## Project Summary
In this project, you will complete the following steps: 
- Train and deploy a model on Sagemaker, using the most appropriate instances. Set up multi-instance training in your Sagemaker notebook.
- Adjust your Sagemaker notebooks to perform training and deployment on EC2.
- Set up a Lambda function for your deployed model. Set up auto-scaling for your deployed endpoint as well as concurrency for your Lambda function.
- Ensure that the security on your ML pipeline is set up properly.

## Step 1: training and deployment on SageMaker
I chose the ml.t3.medium instance in SageMaker for my notebook because it provides sufficient capacity and an optimal balance between CPU and memory resources. When working on machine learning and data analysis tasks, I require efficient performance. Additionally, this instance integrates perfectly with the default kernel of the Data Science 3.0 environment in SageMaker, enhancing productivity and facilitating the development of machine learning models. It also gives me the flexibility to scale up if I need more resources for more complex projects. In summary, this choice ensures a smooth and successful execution of my data science projects.

![notebook_instance](img/notebook_instance.png)



### Step 2: EC2 Training

![ec2](img/ec2.png)

![ec2_evidence](img/ec2_evidence.png)

I chose the g4dn.2xlarge instance type for training my dataset of dog images due to its superior GPU capabilities. The g4dn instances are specifically designed to accelerate machine learning workloads, making them ideal for tasks that involve training deep learning models, such as image classification.

Training a dataset of images can be computationally intensive, especially when using convolutional neural networks (CNNs) or other complex architectures. The g4dn.2xlarge instance is equipped with NVIDIA T4 GPUs, which excel at parallel processing and matrix calculations, significantly speeding up the training process for image datasets.

Moreover, the ample memory and vCPU resources of the g4dn.2xlarge instance provide a good balance for handling large datasets and computational tasks associated with deep learning training. This ensures that the model doesn't run into memory bottlenecks or CPU limitations during the training phase.

In summary, I selected the g4dn.2xlarge instance type for its powerful GPU capabilities and sufficient computational resources, making it well-suited for efficiently training my dog image dataset and helping me achieve faster convergence and accurate models for my image classification task.

#### Compares code from Step 1
1. In the first version of the code, there is no main function that controls the flow of execution. Instead, the train() and test() functions are executed directly after defining them. In the second version, a main() function has been added, which takes command-line arguments and orchestrates the training and evaluation process.

2. In the first version, a fixed number of 5 epochs is set for training, and there is no implementation of early stopping based on validation loss. In the second version, early stopping is introduced to prevent overtraining and improve the efficiency of the model.

3. In the second version, logging commands (logger.info()) have been added to display training and validation information, such as loss and accuracy at each epoch. This helps monitor the training progress and obtain relevant information.

4. In the first version, the loss and accuracy in the test() function are calculated using the floor division operator (//), which performs integer division and returns the result as an integer. This may lead to incorrect results when evaluating models that achieve accuracy that is not an integer. In the second version, the division operator (/) is used, which returns the result as a decimal (floating-point number), which is more appropriate for precision calculations.

5. In the first version, there is a commented line of code (#rom torch_snippets import Report) suggesting that a module called Report can be used to log metrics during training. However, this module is not in use in the current code. In the second version, there are no references to the Report module, suggesting that the intention to use it has been removed.





## Step 3: Lambda function setup
The Lambda function is a Python code designed to be invoked as an AWS Lambda function. Its main purpose is to perform inferences using a PyTorch model hosted on a SageMaker endpoint. The function takes input data from the event and sends it to the SageMaker endpoint to obtain results. It then returns an HTTP response with the inference result in JSON format.

## Step 4: Security and testing

![test_lambda](img/test_lambda.png)
![policies_IAM_lambda](img/policies_IAM_lambda.png)
![dashboard_IAM](img/dashboard_IAM.png)

AWS Lambda and IAM (Identity and Access Management) play crucial roles in securing serverless architectures. However, there are certain security considerations and vulnerabilities that organizations need to be aware of to ensure robust protection. One potential insecurity is the misconfiguration of IAM roles and permissions for Lambda functions. Granting excessive permissions to a Lambda function could lead to unintended access to sensitive resources, exposing the entire AWS environment to potential risks. Additionally, improper handling of IAM roles can lead to privilege escalation attacks, where malicious actors gain unauthorized access to resources beyond their intended scope. Organizations must carefully manage IAM roles associated with Lambda functions, adhering to the principle of least privilege to minimize potential damage in case of a security breach. Regular auditing and monitoring of IAM policies are essential to identify and remediate any security gaps, ensuring a more resilient and secure serverless architecture.

## Step 5: Concurrency and auto-scaling

![concurrency](img/concurrency.png)
![auto-scaling](img/auto-scaling.png)
![auto-scaling-2](img/auto-scaling-2.png)

In concurrency, I have created version 0.0.1 and assigned 3 of the available resources. I have decided to set a scaling target of 30 in both directions (up and down) and a cooldown time of 30 seconds due to the dynamic nature of my application and the expected fluctuations in resource demand. With a target value of 30, I am allowing my environment to automatically increase or decrease the number of available resources based on the current system needs.

The 30-second cooldown time provides a balance between responsiveness and scaling stability. If I were to set a shorter cooldown time, the autoscaling could be overly sensitive to temporary fluctuations, leading to excessive scaling up and down of resources. On the other hand, if I were to set a longer cooldown time, the autoscaling might not respond quickly enough to changes in demand, affecting performance and user experience.

By setting these values, I am ensuring that my application has enough capacity to handle traffic spikes while avoiding unnecessary resource wastage during periods of low demand. This configuration gives me the necessary flexibility to adapt to changes in my application's traffic efficiently and effectively, ensuring an optimal experience for my users and maximizing the utilization of available resources.