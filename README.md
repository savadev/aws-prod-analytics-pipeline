## Data & ML Model Management with PyTorch & AWS

---



Vivek Avlani

*11/18/2018*



****

**Table of Contents**

  * [Production-Scale Data Analytics](#production-scale-data-analytics)
    + [Machine learning in analytics pipelines](#machine-learning-in-analytics-pipelines)
  * [Production Scale Environments](#production-scale-environments)
    + [AWS: A great solution for cloud computing](#aws-a-great-solution-for-cloud-computing)
  * [Managed Services](#managed-services)
  * [Storage Resources](#storage-resources)
    + [Overview of object or file storage resources](#overview-of-object-or-file-storage-resources)
    + [Comparison of important storage resources in AWS:](#comparison-of-important-storage-resources-in-aws)
    + [S3 and RDS: Role within a data analytics pipeline](#s3-and-rds-role-within-a-data-analytics-pipeline)
    + [Redshift: A Storage Service Relevant to Analytics](#redshift-a-storage-service-relevant-to-analytics)
  * [Compute Resources](#compute-resources)
    + [Comparing popular compute options within AWS](#comparing-popular-compute-options-within-aws)
    + [SageMaker: Machine Learning Convenience](#sagemaker-machine-learning-convenience)
  * [Running Machine Learning at Production Scale](#running-machine-learning-at-production-scale)
    + [PyTorch](#pytorch)
    + [Different approaches for saving a model](#different-approaches-for-saving-a-model)
    + [How do we serialize the model: *ONNX*](#how-do-we-serialize-the-model-onnx)
    + [Code Snippet showing ONNX serialization: PyTorch --> ONNX --> MXNet](#code-snippet-showing-onnx-serialization-pytorch----onnx----mxnet)
    + [Running PyTorch on AWS](#running-pytorch-on-aws)
    + [Managing model artifacts on PyTorch deployed through SageMaker](#managing-model-artifacts-on-pytorch-deployed-through-sagemaker)
  * [Deploying PyTorch and other Processing Stages with Docker](#deploying-pytorch-and-other-processing-stages-with-docker)
    + [What is Docker?](#what-is-docker)
    + [Creating custom Docker images for data analytics pipeline](#creating-custom-docker-images-for-data-analytics-pipeline)
    + [AWS Elastic Container Service (ECS): Bringing Docker to AWS](#aws-elastic-container-service-ecs-bringing-docker-to-aws)
  * [Serving Serialized Models](#serving-serialized-models)
    + [Different model server options:](#different-model-server-options)
    + [Process for serving a model using Docker and MXNet Server](#process-for-serving-a-model-using-docker-and-mxnet-server)
  * [Conclusion](#conclusion)
  * [References](#references)

---





### Production-Scale Data Analytics

The main motivation to look into production scale data analytics systems comes when we want to share the results or insights about the work that the analysts have done with the larger team at a firm or the customers of the firm. The work that the analyst has done maybe on his work laptop since he was largely working alone or with a small team of data scientists or analysts. 

The process for sharing the work with the larger team, developing a workflow which replicates the analysis periodically, automating the data ingestion and output, modifying the code to handle distributed computing and optimizing it for speed all falls under the umbrella of production scale data analytics. These steps often involve leveraging cloud solutions to perform some or all of these tasks. Cloud solutions offer a lot of advantages over on-premises solutions which are covered later.

Some of the **properties of a production scale pipeline are**:

- It's highly **scalable** in terms of data, users, computation, and compliance
- It is **performant**
- **Flexible** enough to handle new use cases
- **Economical**
- **Automated** (low or no manual intervention required)
- **Integrated** with other applications or systems
- **Used by more than one person**
- **Tested, validated as updates are made**

#### Machine learning in analytics pipelines

These days, machine learning techniques have become pervasive and analysts often have some sort of learning algorithm as part of the solutions that they have developed. These ML techniques could be used for predicting future outcome, analyzing the effects of existing inputs on some target variable, or even large-scale clustering to generate new insights. All of the new-age data analytics solutions make use of a learning algorithm to derive insights into the data being processes. The days when executives are just satisfied with moving-average sales numbers are gone and they expect 

The image below shows the machine learning life cycle for a typical project:

<img src="https://docs.aws.amazon.com/sagemaker/latest/dg/images/ml-concepts-10.png" style="zoom:90%"/>

It demonstrates the cyclical nature of the project and the need for continuous revisions.

The use of ML in analytics means that when we create production pipelines, we need to handle the problems of using it at scale. 

> **Some of the problems of ML at production-scale:**
>
> - Data science languages like R & Python are often slow
> - Machine learning is compute heavy - often needing GPUs
> - ML Processes are often iterative - they need to be retrained
> - There is a need to prep the data before we train the model
> - Lots of ML specific libraries need to be installed on cluster environment
>

So we need to find ML specific solutions to handle as part of setting up our production analytics pipeline. This means using the ML solutions provided by cloud providers and using the third-party libraries which help us handle the challenges mentioned above.



### Production Scale Environments

To handle these challenges mentioned previously, we need an environment which helps us address some of these problems. We can't run production scale pipelines on our laptop because of many reasons like: the production ***environment needs to be always on***, it needs a ***lot of compute & storage power*** which are limited in personal computers, it needs to be ***able to scale according to the needs of the analytics pipeline***. For e.g., if tomorrow, the data doubles in size, the laptop may not be able to handle it but the production data pipeline has to be robust enough to handle changes in data. We also might decide to ***expose an API which serves predictions*** or results from our final model which means we need a server which is ***capable of handling multiple, concurrent requests***. The work laptop is not an ideal solution for such a use-case. These reasons motivate us to look for other environments which might be able to handle these use-cases.

When we look at production environments, we could either look into **on-premise** hosting of servers, **cloud-based offerings** or a **hybrid solution** which means a part of your environment is hosted on-premise and part is hosted on the cloud solution.

Our main focus is going to be on exploring the cloud offerings by AWS in this paper and we will cover how it can provide utility in creating and maintaining our production pipeline.

> **Advantages of Cloud Computing**:
>
> - Trade capital expense for variable expenses
> - Benefit from economies of scale
> - Stop guessing about capacity
> - Increase development speed and agility
> - Stop spending money running and maintaining data centers
> - Go global in minutes

We have several different **options for cloud-based production scale environments**:
- Amazon: **AWS** (Amazon Web Services)
- Google: **GCP** (Google Cloud Platform)
- Microsoft: **Azure**

<img src="https://www.parkmycloud.com/wp-content/uploads/Makeup-Tutorial-2.jpg" style="zoom:40%"/>

All of these cloud services are very **similar in terms of their base or foundational offerings** and are **similarly priced** to remain competitive in the market.

**AWS is the most mature platform** among the three: it launched in 2006 before Google (2008) and Microsoft's cloud services (2010) but GCP and Azure are catching up fast.

#### AWS: A great solution for cloud computing

**AWS or Amazon Web Services** would make a great choice for a production scale environment because it provides all of the basic services like compute, storage, databases, migration, cloud management,  and security. In addition to these they have over a 100 different auxiliary services with specific use cases whose aim is generally to simplify the process of getting the work started and helps in the maintenance of the application services.

Some of the ***important basic services*** provided by Amazon and their names:

- *Storage*: Amazon **S3** (Object storage)
- *Storage*: Amazon **EBS** (Elastic Block Store)
- *Storage*: Amazon **Glacier** (Low cost, long-term storage for data archiving and backup)
- *Compute*: Amazon **EC2** (Compute servers; provides ability to quickly scale up or down)
- *Database*: Amazon **RDS** (Relational Database Service)
- *Database*: Amazon **DynamoDB** (Fast, flexible NoSQL database service)
- *Network*: Amazon **VPC** (Virtual network to provision logically isolated section of AWS)
- *Security*: AWS **Identity and Access Management** (Securely control access to AWS resources)

And there are many more full-featured services which makes working on the cloud very easy if you know the functionality provided by these services. 

**Overview of AWS Services**

![Image result for aws main services](https://cdn-images-1.medium.com/max/1600/1*U4RqTBdoZ4Sbic4J10Kh3w.png)

### Managed Services

A managed service is any client service that is being managed or maintained by an external entity which hs been contracted by the client. In the context of cloud computing, we can view that spinning up a server is an instance of a managed service where the cloud provider manages the server that is being offered to the client for their work. In this instance the cloud provider is offering **"Infrastructure-as-a-Service"**. 

In the same manner, the client may also offer **"Software-as-a-Service"**: For example, Salesforce offers customer management platforms as SaaS. This means they own the process of maintaining the software and adding features as per the contract as long as the client has subscribed to their service. 

The main ***advantages of a managed service by cloud providers*** like AWS are:

- Client **does not need to maintain the servers**
- Client **does not need to worry about availability** or up-time of the service
- Usually **turns out to be cheaper for customers** due to economies of scale for cloud providers
- Clients **inherit the feature updates and security updates** that the provider pushes
- Client teams **get to spend more time on innovations than support activities**
- **Easier scalability**



### Storage Resources

For running data analytics in a production setting, we often need at least two types of storage resources: **object storage**, **file storage** and **block storage**. These are the basic storage resources that are required in almost every project. Object storage can be used for storing the input files that we are  creating or for storing the outbound files after the process is over. It can also be used to store image or other non-text files which are going to be used in the analytics process. Databases on the other hand can be used to store the structured data output that is created and can also be used as an input source for the analytics pipeline. 

> #### Overview of object or file storage resources 
>
> **(*Important ones in bold*)**
>
> | Service                         | Description                                                  |
> | ------------------------------- | :----------------------------------------------------------- |
> | **Amazon S3**                   | A service that provides scalable and highly durable object storage in the cloud. |
> | Amazon Glacier                  | A service that provides low-cost highly durable archive storage in the cloud. |
> | **Amazon Elastic File System ** | A service that provides scalable network file storage for Amazon EC2 instances. |
> | **Amazon Elastic Block Store**  | A service that provides block storage volumes for Amazon EC2 instances. |
> | Amazon EC2 Instance Storage     | Temporary block storage volumes for Amazon EC2 instances.    |
> | AWS Storage Gateway             | An on-premises storage appliance that integrates with cloud storage. |
> | AWS Snowball                    | A service that transports large amounts of data to and from the cloud. |
> | Amazon CloudFront               | A service that provides a global content delivery network (CDN). |

Although there are a lot of options for storage in AWS, we **often only use one or two services from them** depending on our use-case. For example, if we don't have any data archival needs, Glacier is probably not of much use to us and we can stick to just S3. The table below illustrates the storage resources that are used in a lot of data analytics pipelines.

#### Comparison of important storage resources in AWS:

|                        | S3                                       | RDS                                                          | Redshift                                                     |
| ---------------------- | ---------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Primary Intended Usage | Cheap storage of files                   | Transactional relational database                            | Analytics database for running heavy queries                 |
| Type of data           | Any file/object storage                  | Structured Data (Tables)                                     | Structured Data (Tables)                                     |
| Functionality          | Can only get, delete or put object files | Has the same functionality as a database (can add, delete, retrieve rows   from a particular data source) | Has the same functionality as a database (can add, delete, retrieve rows   from a particular data source) |

#### S3 and RDS: Role within a data analytics pipeline

S3 and RDS are mainly **used as data stores** in an analytics pipeline - we may sometimes use them to **store input files or data and at times, we may store our outputs in them**. 

S3 can also be used as an intermediate layers sometimes to **store the intermediate outputs of our analytics pipeline**. We may choose to store our configuration or model output or predictions as a file in S3. In fact, whenever we want to materialize our findings in the form of a file, S3 is useful.  For example **a trained model, once serialized, can be stored in S3.**

The RDS or other database-based can be used to **publish the results of our analyses.**

#### Redshift: A Storage Service Relevant to Analytics

One of the popular storage services offered by AWS is Redshift which is based on a fork of Postgres but has been ***optimized for performance for querying on large databases.*** It's ideal **use case is as an analytics database** (for example in a data warehousing context). Here we have huge amounts of data from various sources which has been combined in a data warehouse. We want to **run analytical queries on top of this data** and this is where Redshift shines. 

Redshift has been **known to handle petabyte-scale data warehouses** and is competitive compared to even big data technologies like SparkSQL and Impala. It is ideal for users who are comfortable with SQL and want to foray into the realm of big data. There is an upfront cost in terms of provisioning a Redshift instance. 



### Compute Resources

Compute resources provide processing power to our system and enable us to run data analytics. Without compute, we would just be left with storage resources which means all we could do is store and retrieve data from the cloud. Compute resources will enable us to process the data that we are storing in S3 as well as will provide us with the processing power to do auxiliary things like programmatically transferring data, spinning up and spinning down new resources, monitoring services and also orchestrating the data pipeline. Thus, compute resources form a critical part for our data analytics pipeline. 

Examples of some **data processing steps for compute resources:**

- pre-process scripts or queries
- machine learning training code 
- data transfer scripts
- saving the models, testing the model, predicting the model 

All this would all require compute resources like vanilla EC2 or other versions which have the Machine Learning AMIs installed on them.

Resources like EC2, Deep Learning AMIs and SageMaker are all compute resources that are at the heart of any data pipeline - they are the processors in the data pipeline. All the tasks that are performed on the data are performed by these resources. They are the components which have processing power and are capable of executing the scripts we have written. 

#### Comparing popular compute options within AWS

|               | EC2                                                          | LightSail                                 | Amazon ECS                                                   | AWS Lambda                                                   | SageMaker                                                    |
| ------------- | ------------------------------------------------------------ | ----------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Advantages    | General-purposes, scalable server instance                   | Low cost deployment of one or few servers | Running stateless or stateful serverless apps packaged as Docker   containers | Running event-initiated serverless apps that need quick response times | Machine Learning-focused helper performant libraries; built on top of EC2 |
| Disadvantages | Long setup time to install libraries needed for specific use-cases | Not meant to be scalable                  | Tightly integrated with AWS -   difficult to replace with other service | No control over environment                                  | Not as general-purpose as EC2                                |

#### SageMaker: Machine Learning Convenience

Amazon SageMaker is a fully-managed platform that enables developers and data scientists to quickly and easily build, train, and deploy machine learning models at any scale. Amazon SageMaker includes modules that can be used together or independently to build, train, and deploy your machine learning models.

- ***Build***
  - Helps you to quickly connect to training data
  - Select and optimize the best algorithm and framework
  - SageMaker also includes hosted Jupyter notebooks that make it easy to explore and visualize the training data.
  - It includes the most common machine learning algorithms which have been pre-installed and optimized to deliver up to 10 times the performance found anywhere else
  - Pre-configured with TensorFlow, Apache MXNet, and Chainer in Docker containers
- ***Train*** 
  - Single-click to start training the model
  - SageMaker manages all the underlying infrastructure for you and allows you to scale
  - Auto-tuning feature to get the best possible accuracy
- ***Deploy***
  - Helps in deploying the model to production to start generating predictions
  - Works with real-time or batch data
  - Deploys model on auto-scaling clusters of SageMaker ML instances



The following image shows a simplistic use-case where we batch-process data and get inferences from a SageMaker instance which is part of a larger cluster of machine learning instances:

![Image result for running pytorch on aws workflow](https://docs.aws.amazon.com/sagemaker/latest/dg/images/batch-transform.png)





### Running Machine Learning at Production Scale

The basic machine learning workflow consists of 2 separate processes: one for training, testing and creating the model. The other one is for running in a production when you want to make inferences or predictions for a test dataset. This test or live data can be misclassified by our model and then it is fed back to the training data after it's been correctly labeled. Then we train the model again and this is how our model gets better.



![1542665751249](C:\Users\Vivek\AppData\Roaming\Typora\typora-user-images\1542665751249.png)



***Training and inference workflows should be separate*** and not sequential because **they can occur independently of one another** once the first model has been trained. This means that we **don't create a dependency** between the two flows. For running the inference part, we don't need to train the model again. This results in just running the part of the pipeline which is necessary and **reduces run time**. Separating out the inference part also improves the performance of the system since we can **spin up other resources for just the inference** or model serving part.

#### PyTorch

It's a python based scientific computing package which can be used as:

- A replacement for NumPy which **uses the power of GPUs**
-  A **deep learning research platform** that provides maximum flexibility and speed
- It is known for providing two of the most high-level features; namely, **tensor computations with strong GPU acceleration support and building deep neural networks** on a tape-based autograd systems.

Some of the ***advantages that PyTorch and TensorFlow have over traditional machine learning libraries*** are that they are **optimized to work in a distributed environment**. They are **designed for speed and efficient execution**. They can **benefit from the presence of GPUs** and can train models much more quickly than libraries like scikit-learn. These libraries are often used for training and testing **deep learning models like neural nets**.

#### Different approaches for saving a model

The process of saving a model to disk is called model serialization. There are two main approaches for serializing and restoring a model in PyTorch:

The first (recommended) **saves and loads only the model parameters or artifacts:**

```python
torch.save(the_model.state_dict(), PATH)
```

Then later to load the model in memory again:

```python
the_model = TheModelClass(*args, **kwargs)
the_model.load_state_dict(torch.load(PATH))
```

The second **saves and loads the entire model:**

```python
torch.save(the_model, PATH)
```

Then later to load the model in memory:

```python
the_model = torch.load(PATH)
```

However in this case, the serialized data is bound to the specific classes and the exact directory structure used, so it can break in various ways when used in other projects, or after some serious refactors.

#### How do we serialize the model: *ONNX*

**PyTorch** has integrations which allow it to save the model (or serialize the model) as an ONNX model. **ONNX is an inter-operable model serialization format** which allows us to transfer a trained model from one library to another. Thus, it is useful that PyTorch allows us to save and load ONNX models. This makes it interoperable with other Machine Learning libraries like MXNet.

ONNX supports PyTorch, MXNet, Chainer, Caffe2 and many other deep learning libraries (see image below). It is gaining popularity and is one of the preferred choices when we think about serializing the final model. It saves the different model artifacts based on the type of model and can be read and saved by all the supported libraries or frameworks. 

**Supported Frameworks**

![1542587350200](C:\Users\Vivek\AppData\Roaming\Typora\typora-user-images\1542587350200.png)

#### Code Snippet showing ONNX serialization: PyTorch --> ONNX --> MXNet

The following code snippets show how to save a model in PyTorch using ONNX and loading it again in MXNet:

Create a new file with your text editor, and use the following program in a script to train a mock model in PyTorch, then export it to the ONNX format.

```python
# Build a Mock Model in PyTorch with a convolution and a reduceMean layer
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.onnx as torch_onnx

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3,3), stride=1, padding=0, bias=False)

    def forward(self, inputs):
        x = self.conv(inputs)
        #x = x.view(x.size()[0], x.size()[1], -1)
        return torch.mean(x, dim=2)

# Use this an input trace to serialize the model
input_shape = (3, 100, 100)
model_onnx_path = "torch_model.onnx"
model = Model()
model.train(False)

# Export the model to an ONNX file
dummy_input = Variable(torch.randn(1, *input_shape))
output = torch_onnx.export(model, 
                          dummy_input, 
                          model_onnx_path, 
                          verbose=False)
print("Export of torch_model.onnx complete!")
```

After you run this script, you will see the newly created .onnx file in the same directory. Now, switch to the MXNet Conda environment to load the model with MXNet.

Create a new file with your text editor, and use the following program in a script to open ONNX format file in MXNet.

```python
import mxnet as mx
from mxnet.contrib import onnx as onnx_mxnet
import numpy as np

# Import the ONNX model into MXNet's symbolic interface
sym, arg, aux = onnx_mxnet.import_model("torch_model.onnx")
print("Loaded torch_model.onnx!")
print(sym.get_internals())
```

After you run this script, MXNet will have loaded the model, and will print some basic model information.

#### Running PyTorch on AWS

There are **different ways to run PyTorch on AWS**. Some are more complicated than others.

1. Spin up a **vanilla EC2 instance and install PyTorch**: This approach involves the most amount of work but is resistant to cloud vendor lock-in
2. Spin up a vanilla EC2 and run a Docker container off of a **PyTorch Docker image** 
3. Use one of **Amazon Deep Learning AMIs** which has PyTorch installed already - all you need to do is launch an EC2 with one of these AMIs
4. Third option which is very convenient is using **SageMaker** which provides a fully-managed machine learning platform

#### Managing model artifacts on PyTorch deployed through SageMaker

Managing the model artifacts is easy using PyTorch through SageMaker - we just need to specify the directory to which we want to save the model artifacts:

In order to save your trained PyTorch model for deployment on SageMaker, your training script should save your model to a certain filesystem path called `model_dir`. This value is accessible through the environment variable `SM_MODEL_DIR`. The following code demonstrates how to save a trained PyTorch model named `model` as `model.pth` at the :

```python
import argparse
import os
import torch

if __name__=='__main__':
    # default to the value in environment variable `SM_MODEL_DIR`. Using args makes the script more portable.
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    args, _ = parser.parse_known_args()

    # ... train `model`, then save it to `model_dir`
    with open(os.path.join(args.model_dir, 'model.pth'), 'wb') as f:
        torch.save(model.state_dict(), f)
```

After your training job is complete, SageMaker will compress and upload the serialized model to S3, and your model data will be available in the S3 `output_path` you specified when you created the PyTorch Estimator.

The easiest way to deploy PyTorch on AWS is to use the SageMaker instances but it comes with the cost of SageMaker added to your EC2 instance. The following diagram illustrates how we can leverage PyTorch in the training and subsequent phases:

![Image result for pytorch sagemaker diagram](https://docs.aws.amazon.com/sagemaker/latest/dg/images/sagemaker-architecture.png)





### Deploying PyTorch and other Processing Stages with Docker

Before diving into how we can deploy PyTorch with Docker, let's understand what Docker is in the first place.

#### What is Docker?

Docker is a tool for creating, deploying, and running applications by using containers. Container allows a developer to package up an application with all the parts it needs like libraries and other dependencies, and ship it all out as one package. If you do this, then you make sure that it will run on any Linux machine regardless of any customized settings that might have that could differ from the developer's machine which was used for developing and testing.

Docker is a bit like a virtual machine but instead of creating whole operating system, it allows applications to use the same Linux kernel as the system that they are running on.

![monolith_2-VM-vs-Containers](https://d1.awsstatic.com/Developer%20Marketing/containers/monolith_2-VM-vs-Containers.78f841efba175556d82f64d1779eb8b725de398d.png)

Building and deploying new applications is faster with containers. Docker containers wrap up software and its dependencies into a standardized unit for software development that includes everything it needs to run: code, runtime, system tools and libraries. This guarantees that your application will always run the same and makes collaboration as simple as sharing a container image.

> **Advantages of using Docker**
>
> - Ship more software faster
> - Seamless transfer from dev to test to production envrionments
> - Easier to run more code on each server which improves utilization and helps to save money
> - Standardized operations

Dockerized containers help us in recreating replicas of the development environment in test and production systems. This helps avoid the problems of version incompatibility of libraries in different environments. The dockerized container has the same binaries and libraries for the different applications. 

Thus it standardizes the workflow and smoothens the process of transferring work from development to operations. 

#### Creating custom Docker images for data analytics pipeline

There are 2 ways of creating our own Docker images that we can then deploy into AWS using ECS. 

Creating Docker images:

> In general, there are **two ways to create a new Docker image**:
>
> - Create an image from an existing image - we can customize the existing image with the changes we want and create the new image
> - Use a Dockerfile which is a  file of instructions to specify the base image and the changes you want to make to it.

There are various services within AWS that help in managing Docker containers including Elastic Container Service, Elastic Kubernetes Service, and AWS Batch. 

We will be focusing on ECS here which provides orchestration service for Docker containers within the AWS environment.

#### AWS Elastic Container Service (ECS): Bringing Docker to AWS

AWS offers ECS which helps with a lot of the management of selection, and creating instances with specific docker images already installed on them. It allows you to use a container registry which could either be AWS Elastic Container Registry or any other registry like DockerHub. 

You of course also have the option of launching a vanilla EC2 and installing Docker on that machine and getting the Docker image that you want from whichever registry and running it as a container.

But, ECS allow you to containerize all sorts of your applications from long-running apps to microservices. ECS is deeply integrated with AWS and thus provides with a complete solution for building and running a wide range of containerized applications. 

![product-page-diagram_ECS_1](https://d1.awsstatic.com/diagrams/product-page-diagrams/product-page-diagram_ECS_1.86ebd8c223ec8b55aa1903c423fbe4e672f3daf7.png)



### Serving Serialized Models

After we have finished training the model and evaluated the results, we serialize the final model that we obtain so that it can be used to serve predictions for new input data. The serialization of the model can take several forms including using library-specific serialization techniques or new library-agnostic serialization protocols. These new age interoperable AI tools help us move seamlessly across different libraries that maybe used.

There are 2 options for serving serialized models:

- We could do a **batch process** to get the results
- We could also setup an **API-based service **which gives results based on user API calls

<img src="C:\Users\Vivek\AppData\Roaming\Typora\typora-user-images\1542666703982.png" style="zoom:70%"/>

API-based model servers are useful when we want to serve a small number of discrete requests that come in from the users. It makes it convenient for them to quickly get the predictions for an input and act on it. They are not meant to handle hundreds or thousands of consecutive requests from the users.  It would slow down as the number of requests increases.

If the number of requests are large and the predictions are required on a batch of inputs then we should consider using the batch process for serving the predictions or outputs from the model. While this may not be as easy to use as an API, for a large number of requests, it would be much faster than an API-based server where you would send a single request for each observation in your input. 

#### Different model server options:

- TensorFlow Serving
- Clipper
- Model Server for Apache MXNet
- DeepDetect
- TensorRT

The table below ***summarizes the differences in model servers***:

|               | TensorFlow Serving     | Clipper                                   | Model Server for Apache MXNet (MMS)                          |
| ------------- | ---------------------- | ----------------------------------------- | ------------------------------------------------------------ |
| Advantages    | High performance       | Easier to integrate with production stack | Automated setup - easy to integrate; ability to package custom processing   code |
| Disadvantages | Difficult to integrate | Less mature than other frameworks         | Not as performant as TensorFlow Serving                      |

#### Process for serving a model using Docker and MXNet Server

The image below shows how we could deploy our model server using Docker images which are being  managed by the AWS ECS Service. The process has been described below the diagram.

<img src="C:\Users\Vivek\AppData\Roaming\Typora\typora-user-images\1542768054867.png" style="zoom:70%"/>

Once you have trained a model and validated that it works, you would like to ideally put into production for your users so that they can make API calls to it and get the output. To do this, we first need to decide which server we are going to use. There are several options as given in the table above. We will decide that we are going to go ahead with an MXNet Server. We need to first use a Docker image which contains MXNet and MXNet Server. Once we have this we can deploy it using Docker engine as a container. We need to configure the server so that it points to the serialized model that we have stored (possibly in our S3 environment). This code which sets up the connection to connect the Server to the serialized model could again reside in our Docker image. 

### Conclusion

We have now explored how we can setup a machine learning data analytics pipeline, leveraging the cloud infrastructure and services provided by AWS. We now appreciate the complexity of the challenge and can think through a solution when presented with such a problem. There are lots of different components that are needed to develop a production pipeline including hardware infrastructure like storage resources, compute resources and software resources which are available as open-source projects. 

The success and performance of a data analytics pipeline hinges heavily on how well the developer understands the use-cases and pros-cons of various technology stacks. There are lots of services available which makes choosing the right one for the job very challenging.

The need for distributed computing, due to the size of the data, makes it necessary to use deep-learning and machine learning frameworks like TensorFlow and PyTorch since traditional ML libraries like scikit-learn don't support parallelism and multi-core computing out-of-the-box. The installation of these libraries is not a trivial task and deploying the solution into production is even a bigger challenge. This introduces the need for containers and Docker comes into the picture. We use Docker to aid us in the deployment and testing of our ML models. It allows us to separate out our processes and handles dependencies in an elegant way which makes our code flexible and more robust.

The final models then need to be opened up in production so that they can start serving requests. Creating model-serving pipelines is another non-trivial exercise and instead of coming up with our own solution from scratch, we rely on open-source model serving libraries like MXNet Server or TensorFlow Serving. These frameworks greatly reduce our production deployment time. We could also not go the API-based route and rely on batch-processing for serving model outputs depending on the business use-case.

The things covered here gives us an idea of the challenges involved and provide a way out of our problems through the use of correct technology stacks and the correct setup code. Although the problem of setting up a production pipeline is still a daunting one, this gives us an idea of how we might go about tackling it.



### References

1. *https://pytorch.org/docs/stable/notes/serialization.html#recommended-approach-for-saving-a-model*

2. *https://github.com/aws/sagemaker-pytorch-container*

3. *https://d1.awsstatic.com/whitepapers/Storage/AWS%20Storage%20Services%20Whitepaper-v9.pdf*

4. https://github.com/awslabs/mxnet-model-server

5. https://pytorch.org/tutorials/

6. https://pytorch.org/docs/stable/onnx.html

7. https://d1.awsstatic.com/whitepapers/Storage/AWS%20Storage%20Services%20Whitepaper-v9.pdf

8. https://aws.amazon.com/products/compute/?nc2=h_m1

9. https://aws.amazon.com/products/databases/?nc2=h_m1

10. https://medium.com/@vikati/the-rise-of-the-model-servers-9395522b6c58

11. https://docs.aws.amazon.com/dlami/latest/devguide/tutorial-onnx-pytorch-mxnet.html

12. https://onnx.ai/about
