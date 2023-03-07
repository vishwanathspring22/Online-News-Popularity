# Machine Learning with Business Analytics
## Title: Given the features prediction of article popularity 
### The data set is extracted from UCI machine learning directory and the data set 
### is called as 'Online News Popularity'. This dataset is extracted from Mashable website
### Kelwin Fernandes, Pedro Vinagre, Paulo Cortez and Pedro Sernadela web scraped the data from Mashable webiste
### Acquisition Date: January 8, 2015
### Acquisition Period : January 5th 2013 to January 7th 2015 

## Install The packages
install.packages("dplyr")
install.packages("caret")
install.packages("caTools")
install.packages("ggplot2")
install.packages("lares")
install.packages("psych")
install.packages("boruta")
install.packages("party")
install.packages("data.table")
install.packages("skimr")
install.packages("tidyr")
install.packages("car")
install.packages("magrittr")
install.packages("Boruta")
install.packages("naivebayes")
install.packages("e1071")
install.packages("mlbench")

## Load the libraries
library(dplyr)
library(caret)
library(caTools)
library(ggplot2)
library(lares)
library(psych)
library(Boruta)
library(party)
library(data.table)
library(skimr)
library(tidyr)
library(car)
library(magrittr)
library(dplyr)
library(tidyr)
library(Boruta)
library(naivebayes)
library(e1071)
library(mlbench)


#******************************************************************
# Descriptive Analysis
#******************************************************************

#Import the dataset
data <- read.csv("OnlineNewsPopularity.csv")


#Let us the view the dataset
View(data)

#There are 39644 instances and 61 attributes in the dataset and the target attribute is shares
dim(data)

#Summary 
summary <- skim(data)
print(summary)
summary(data)

#Remove the duplicates from the data frame using distinct function
d1<- distinct(data)
dim(d1)

#Through Descriptive analysis and by experience let us try remove attributes
#Since the number is high, computational expenses are high 
#Non-value added variables can be removed and let us continue with that process

#Elimination of non-value added and non-predictive variables from the dataset
#Non-predictive is said in the source also

d1<- subset(d1, select = -c(url, timedelta))

#From the summary, we can see that this value has highest as 1042 and others are almost 1

boxplot(d1$n_non_stop_words)
d1 <- subset(d1, d1$n_non_stop_words !=1042)
#Even after seeing the summary this attribute seems to have errors
#Almost every value looks to be 1
#We can remove this attribute
d1<- subset(d1, select = -n_non_stop_words)

summary(d1)
dim(d1)

#We observe that another outlier is removed from n_unique_tokens
#It had an maximum value of 701.0 and that was also removed 

#We will remove LDA related attributes since there is no info about them 
#There is no source of clear info in website of UCI

d2<- subset(d1, select = -c(LDA_00, LDA_01, LDA_02, LDA_03, LDA_04))
dim(d2)

#Now from 61 attributes we have reduced it to 53 attributes

############################################
  
View(d2)
#From the current dataset, we can see that weekend attribute and all weekdays is mentioned
#To make further simplification, let us keep weekend attribute eliminating others
#Weekend attribute signifies whether the target attribute is effects whether the share is on weekday or weekend 
#In the column, "0" signifies the share is on weekday and "1" signifies on weekend

d3 <- subset(d2, select = -c(weekday_is_monday,
                                     weekday_is_wednesday,
                                     weekday_is_thursday,
                                     weekday_is_friday,
                                     weekday_is_tuesday,
                                     weekday_is_saturday, 
                                     weekday_is_sunday))

ggplot(d2, aes(x = weekday ,fill=weekday)) +
  geom_bar(show.legend = T) +
  theme_minimal() +
  labs(title = 'Number of Shares on Different Weekday', 
       x = 'Weekday', y = 'Number of Shares') +
  theme(plot.title = element_text(hjust = 0.5), plot.subtitle = element_text(hjust = 1))

dim(d3)
#The dimension has further reduced to 46 
str(d3)
#We see that weekend variable is numerical datatype and it has to be converted into factor
#To consider that "0" for weekday and "1", the model has to understand, so it is converted to factor

d3$is_weekend <- as.factor(d3$is_weekend)
str(d3$is_weekend)

#It is converted to factor

#############################################
  
#Let us consider channel related attributes in the dataset
#Let us view the columns related using dplyr package and select() function

select(d3, c(11:16))

#We see that it is hot-encoded and represented by "0s" and "1s"
#Let us convert into one column and making it easier to visualize 
#It helps in reducing the number of attributes and easy representataion

#channel creation
d3$channel <-  as.numeric(d3$data_channel_is_lifestyle) * 1 + 
  as.numeric(d3$data_channel_is_entertainment) * 2 + 
  as.numeric(d3$data_channel_is_bus) * 3 + 
  as.numeric(d3$data_channel_is_socmed) * 4 + 
  as.numeric(d3$data_channel_is_tech) * 5 + 
  as.numeric(d3$data_channel_is_world) * 6 

(for(i in 1:nrow(d3))
{
  if(d3[i,]$channel == 0)
  {
    d3[i,]$channel = 7
  }
}
)

d3$channel <- factor(d3$channel, levels = 1:7, labels = c("Lifestyle", "Entertainment", "Business", "Social_Media", "Technology", "World", "others"))

View(d3)
d4 <- subset(d3, select = -c(data_channel_is_lifestyle,data_channel_is_entertainment, data_channel_is_bus, data_channel_is_socmed, data_channel_is_tech, data_channel_is_world))

dim(d4)
#The dimension has further reduced to 41

str(d4)
#We observe that channel is a factor, so that model will understand that it is categorical
# Creating Numeric Format
d4$channel <- as.numeric(d4$channel)
d4$is_weekend <- as.numeric(d4$is_weekend)
#######################################################


#*********************
#*Correlation Analysis
#*********************

#After exploring the dataset and removing few attributes through redundancy
#Let us explore about correlation about inter-relation between independent and dependent variables
#Let us assign a new dataset for this function

cor_data<- d4
str(cor_data)
cor_data$channel <- as.numeric(cor_data$channel)
cor_data$shares <- as.numeric(cor_data$shares)
cor_data$is_weekend <- as.numeric(cor_data$is_weekend)
str(cor_data)
cc <- cor(cor_data)


dev.off()
corPlot(cor_data)
warnings()

dev.off()
corr_cross(cor_data,
           max = 0.9, # name of dataset
           max_pvalue = 0.05, # display only significant correlations (at 5% level)
           top = 25  # display top 10 couples of variables (by correlation coefficient)
)

#We observe variables which exhibit multi-collinearity
#Let us confirm out the variables with multi-collinearity through Variance Inflation Factor
#Setting the VIF threshold as 10 based on a journal reference

#To find out, let us build a linear model with all variables

model <- lm(shares ~ ., data = cor_data)
summary(model)

vif(model)

#create vector of VIF values
vif_values <- vif(model)

View(vif_values)

#create horizontal bar chart to display each VIF value
barplot(vif_values, main = "VIF Values", horiz = TRUE, col = "steelblue")

#add vertical line at 5
abline(v = 10, lwd = 3, lty = 2)

#From the graph and vif model values, we observe that there are 8 attributes 
#Which has more than 10 VIF

#Let us create a dataset which eliminates these variables

d5 <- subset(d4, select = -c(kw_avg_min, n_unique_tokens, n_non_stop_unique_tokens, average_token_length,
                             kw_max_min, self_reference_avg_sharess, rate_positive_words, rate_negative_words))
str(d5)
dim(d5)

################################################################### Exploratory Analysis - start ##########################################################################
# Non-logged and Logged visualization of target variables.
dim(d5)
d5[,33] <- as.numeric(d5[,33])
par(mfrow=c(3,4))
for(i in 1:length(d5)){hist(d5[,i], xlab=names(d5)[i])}
options("install.lock"=FALSE)
install.packages("qqplotr")
library(qqplotr)
ggplot(mapping = aes(sample = d5$shares))+
  stat_qq_point(size = 2,color = "red")+
  stat_qq_line(color="green")+
  labs(title = 'Probability Plot')+
  xlab("Theoretical Quantities") + ylab("Ordered Values")

nonLog_shares <- d5$shares
hist(nonLog_shares)
dev.off()

# Taking log of target variable to reduce the skewness. so that we can get more accuracy.
log_shares <- log(nonLog_shares)
hist(log_shares)
ggplot(mapping = aes(sample = log_shares))+
  stat_qq_point(size = 2,color = "red")+
  stat_qq_line(color="green")+
  labs(title = 'Probability Plot')+
  xlab("Theoretical Quantities") + ylab("Ordered Values")

visualization_data <- d5
visualization_data <- as.data.frame(visualization_data)
str(d5)
visualization_data$Popularity <- ifelse(d5$shares > 1400, 1, 0)
visualization_data$Popularity <- as.factor(visualization_data$Popularity)
visualization_data$Popularity <- factor(visualization_data$Popularity, levels = 0:1, labels = c("not popular", "popular"))
visualization_data$Popularity

ggplot(visualization_data,aes(x=Popularity, fill=Popularity))+
  geom_bar(show.legend = T)+
  theme_minimal()+
  labs(title = "Popularity of article",
       x= "Popularity", y= "Number of articles")+
  theme(plot.title = element_text(hjust = 0.5), plot.subtitle = element_text(hjust = 1))
table(visualization_data$Popularity)
# As we can see from the above plot that our dataset is balanced already. so there is no need to balance the dataset

d6<-d5
keyword<-subset(d6,select=c(kw_max_avg,kw_min_max,kw_min_min, kw_min_avg, kw_avg_max, kw_avg_avg,shares))

############################################################## Visualization ############################################################################
# Number of images and share.
ggplot(d5, aes(x = log_shares, y = num_imgs, color="orange")) +
  geom_point(show.legend = T) +
  theme(text = element_text(size=50),
        axis.text.x = element_text(angle=90, hjust=1)) +
  theme_minimal() +
  labs(title = 'Number of shares vs number of images in the news article', 
       x = 'Number of Images', y ='Number of Shares') + 
  theme(plot.title = element_text(hjust = 1.0), plot.subtitle = element_text(hjust = 5))

# In popular news article, best keywords are having maximum number of shares.
log_kw_maxmax<- log(d5$kw_max_max)
ggplot(keyword, aes(x = log_kw_maxmax, y = log_shares, color= "red")) +
  geom_point(show.legend = T) +
  theme_minimal() +
  labs(title = 'Maximum number of best keywords vs number of shares', 
       x = 'KW_MAX_MAX', y ='Number of Shares') + 
  theme(plot.title = element_text(hjust = 1), plot.subtitle = element_text(hjust = 1))

# In Popular new, average keyword shares lies between 8 to 15. which means, if the average keyword is having more number of shares. That news is not likely to be popular.
log_kw_maxavg<-log(d5$kw_max_avg)
ggplot(keyword, aes(x = log_kw_maxavg, y = log_shares, color="red")) +
  geom_point(show.legend = T) +
  theme_minimal() +
  labs(title = 'Maximum number of average keywords vs number of shares', 
       x = 'KW_MAX_AVG', y ='Number of Shares') + 
  theme(plot.title = element_text(hjust = 0.5), plot.subtitle = element_text(hjust = 1))
#the number of shares for articles with 8-10 keywords is maximum

#Polarity and shares
ggplot(d5, aes(x = min_positive_polarity, y = log_shares)) +
  geom_bar(stat="identity",fill="blue") +
  theme_minimal() +
  labs(title = 'Minimum positive polarity vs number of shares', 
       x = 'Minimum positive polarity', y ='Number of Shares') +
  theme(plot.title = element_text(hjust = 0.5), plot.subtitle = element_text(hjust = 1))
# The highest shares are for a minumum polarity between 0 to 0.25

ggplot(d5, aes(x = max_positive_polarity, y = log_shares)) +
  geom_bar(stat="identity",fill="red") +
  theme_minimal() +
  labs(title = 'Maximum positive polarity vs number of shares', 
       x = 'maximum positive polarity', y ='Number of Shares') +
  theme(plot.title = element_text(hjust = 1), plot.subtitle = element_text(hjust = 1))
# Number of shares is highest for a maximum polarity of 1. Which may be like news article with good positive emotions are more likely to get more number of shares.

ggplot(d5, aes(x = min_negative_polarity, y = log_shares)) +
  geom_bar(stat="identity",fill="red") +
  theme_minimal() +
  labs(title = 'Minimum negative polarity vs number of shares', 
       x = 'Minimum negative polarity', y ='Number of Shares') +
  theme(plot.title = element_text(hjust = 1), plot.subtitle = element_text(hjust = 1))

ggplot(d5, aes(x = max_negative_polarity, y = log_shares)) +
  geom_bar(stat="identity",fill="red") +
  theme_minimal() +
  labs(title = 'Maximum negative polarity vs number of shares', 
       x = 'Maximum negative polarity', y ='Number of Shares') +
  theme(plot.title = element_text(hjust = 1), plot.subtitle = element_text(hjust = 1))

# If the negative polarity is high, the news are more likely to get spreaded and more number of shares.

#Number of keywords
ggplot(d5, aes(x = num_keywords, y = log_shares)) +
  geom_bar(stat="identity",fill="green") +
  theme_minimal() +
  labs(title = 'Number of keywords versus number of shares', 
       x = 'number of keywords', y ='Number of Shares') +
  theme(plot.title = element_text(hjust = 1), plot.subtitle = element_text(hjust = 1))
# articles with keywords more than 5 have more shares 

#N tokens content 
ggplot(d5, aes(x = n_tokens_content, y = log_shares, color="red")) +
  geom_point(show.legend = T) +
  theme_minimal() +
  labs(title = 'number of words in the news vs number of shares', 
       x = 'n_tokens_content', y ='Number of Shares') +
  theme(plot.title = element_text(hjust = 1), plot.subtitle = element_text(hjust = 1))
# articles with more words do not receive a lot of shares


ggplot(d5, aes(x = num_videos, y = log_shares)) +
  geom_bar(stat="identity",fill="blue") +
  theme_minimal() +
  labs(title = 'Number of videos vs number of shares  ', 
       x = 'number of videos', y ='Number of Shares') +
  theme(plot.title = element_text(hjust = 0.5), plot.subtitle = element_text(hjust = 1))
# articles with 0 videos have most number of shares

############################################################################ Visualization ##############################################################################

################################################################### Exploratory Analysis - end ##########################################################################
#The dataset has further reduced to 29 variables 
#From this dataset, we can further proceed with outlier detection

################################################### Box Plot for all the variables - start

# Box plot for first 15 variables.
box_without_Scaling_1 <- d5 %>%
  gather(variable,values,1:11)
options(repr.plot.width = 8,
        repr.plot.height = 14)

ggplot(box_without_Scaling_1)+
  geom_boxplot(aes(x=variable,y=values),fill="cadetblue") +
  facet_wrap(~variable,ncol=8,scales="free")+
  theme(strip.text.x = element_blank(),
        text = element_text(size=14))

box_without_Scaling_2 <- d5 %>%
  gather(variable,values,12:21)
options(repr.plot.width = 8,
        repr.plot.height = 14)

ggplot(box_without_Scaling_2)+
  geom_boxplot(aes(x=variable,y=values),fill="cadetblue") +
  facet_wrap(~variable,ncol=8,scales="free")+
  theme(strip.text.x = element_blank(),
        text = element_text(size=14))

# Box plot for second set of variables.
box_without_Scaling_3 <- d5 %>%
  gather(variable,values,22:29)
options(repr.plot.width = 8,
        repr.plot.height = 14)

ggplot(box_without_Scaling_3)+
  geom_boxplot(aes(x=variable,y=values),fill="cadetblue") +
  facet_wrap(~variable,ncol=8,scales="free")+
  theme(strip.text.x = element_blank(),
        text = element_text(size=14))

# after plotting the graph, we are keeping only maximum shares of best, average and worst keywords. which means the keywords which are having max shares in them.
d5 <- subset(d5, select = -c(kw_min_min, kw_max_avg, kw_avg_avg,
                                kw_min_avg))

dim(d5)

################################################### Box Plot for all the variables - end

# #<!-------------------------- average_token_length ------------------------->
#   Q <- quantile(onp$average_token_length, probs =c(0.25, 0.75), na.rm = FALSE)
# iqr <- IQR(onp$average_token_length)
# up <- Q[2] + 1.5*iqr  #Upper Range
# low <- Q[1] - 1.5*iqr #Lower range 
# onp <- subset(onp, onp$average_token_length > (Q[1] - 1.5*iqr) & onp$average_token_length < (Q[2]+1.5*iqr))
# 
# <!-------------------------- kw_avg_min ------------------------->
#   Q <- quantile(onp$kw_avg_min, probs =c(0.25, 0.75), na.rm = FALSE)
# iqr <- IQR(onp$num_self_hrefs)
# up <- Q[2] + 1.5*iqr  #Upper Range
# low <- Q[1] - 1.5*iqr #Lower range 
# onp <- subset(onp, onp$kw_avg_min > (Q[1] - 1.5*iqr) & onp$kw_avg_min < (Q[2]+1.5*iqr))
# 
# <!-------------------------- kw_max_min ------------------------->
#   Q <- quantile(onp$kw_max_min, probs =c(0.25, 0.75), na.rm = FALSE)
# iqr <- IQR(onp$kw_max_min)
# up <- Q[2] + 1.5*iqr  #Upper Range
# low <- Q[1] - 1.5*iqr #Lower range 
# onp <- subset(onp, onp$kw_max_min > (Q[1] - 1.5*iqr) & onp$kw_max_min < (Q[2]+1.5*iqr))
# 
# <!-------------------------- kw_min_max ------------------------->
#   Q <- quantile(onp$kw_min_max, probs =c(0.25, 0.75), na.rm = FALSE)
# iqr <- IQR(onp$kw_min_max)
# up <- Q[2] + 1.5*iqr  #Upper Range
# low <- Q[1] - 1.5*iqr #Lower range 
# onp <- subset(onp, onp$kw_min_max > (Q[1] - 1.5*iqr) & onp$kw_min_max < (Q[2]+1.5*iqr))
# 
# <!-------------------------- kw_min_min ------------------------->
#   Q <- quantile(onp$kw_min_min, probs =c(0.25, 0.75), na.rm = FALSE)
# iqr <- IQR(onp$kw_min_min)
# up <- Q[2] + 1.5*iqr  #Upper Range
# low <- Q[1] - 1.5*iqr #Lower range 
# onp <- subset(onp, onp$kw_min_min > (Q[1] - 1.5*iqr) & onp$kw_min_min < (Q[2]+1.5*iqr))
# 
# <!-------------------------- n_tokens_content ------------------------->
#   Q <- quantile(onp$n_tokens_content, probs =c(0.25, 0.75), na.rm = FALSE)
# iqr <- IQR(onp$n_tokens_content)
# up <- Q[2] + 1.5*iqr  #Upper Range
# low <- Q[1] - 1.5*iqr #Lower range 
# onp <- subset(onp, onp$n_tokens_content > (Q[1] - 1.5*iqr) & onp$n_tokens_content < (Q[2]+1.5*iqr))
# dim(onp)
# 
# <!-------------------------- num_hrefs ------------------------->
#   Q <- quantile(onp$num_hrefs, probs =c(0.25, 0.75), na.rm = FALSE)
# iqr <- IQR(onp$num_hrefs)
# up <- Q[2] + 1.5*iqr  #Upper Range
# low <- Q[1] - 1.5*iqr #Lower range 
# onp <- subset(onp, onp$num_hrefs > (Q[1] - 1.5*iqr) & onp$num_hrefs < (Q[2]+1.5*iqr))
# 
# <!-------------------------- num_self_hrefs ------------------------->
#   Q <- quantile(onp$num_self_hrefs, probs =c(0.25, 0.75), na.rm = FALSE)
# iqr <- IQR(onp$num_self_hrefs)
# up <- Q[2] + 1.5*iqr  #Upper Range
# low <- Q[1] - 1.5*iqr #Lower range 
# onp <- subset(onp, onp$num_self_hrefs > (Q[1] - 1.5*iqr) & onp$num_self_hrefs < (Q[2]+1.5*iqr))
# 
# <!-------------------------- num_videos ------------------------->
#   Q <- quantile(onp$num_videos, probs =c(0.25, 0.75), na.rm = FALSE)
# iqr <- IQR(onp$num_videos)
# up <- Q[2] + 1.5*iqr  #Upper Range
# low <- Q[1] - 1.5*iqr #Lower range 
# onp <- subset(onp, onp$num_videos > (Q[1] - 1.5*iqr) & onp$num_videos < (Q[2]+1.5*iqr))
# 
# <!-------------------------- n_non_stop_unique_tokens ------------------------->
#   Q <- quantile(onp$n_non_stop_unique_tokens, probs =c(0.25, 0.75), na.rm = FALSE)
# iqr <- IQR(onp$n_non_stop_unique_tokens)
# up <- Q[2] + 1.5*iqr  #Upper Range
# low <- Q[1] - 1.5*iqr #Lower range 
# onp <- subset(onp, onp$n_non_stop_unique_tokens > (Q[1] - 1.5*iqr) & onp$n_non_stop_unique_tokens < (Q[2]+1.5*iqr))
# 
# <!-------------------------- n_unique_tokens ------------------------->
#   Q <- quantile(onp$n_unique_tokens, probs =c(0.25, 0.75), na.rm = FALSE)
# iqr <- IQR(onp$n_unique_tokens)
# up <- Q[2] + 1.5*iqr  #Upper Range
# low <- Q[1] - 1.5*iqr #Lower range 
# onp <- subset(onp, onp$n_unique_tokens > (Q[1] - 1.5*iqr) & onp$n_unique_tokens < (Q[2]+1.5*iqr))
# 
# <!-------------------------- n_tokens_title ------------------------->
#   Q <- quantile(onp$n_tokens_title, probs =c(0.25, 0.75), na.rm = FALSE)
# iqr <- IQR(onp$n_tokens_title)
# up <- Q[2] + 1.5*iqr  #Upper Range
# low <- Q[1] - 1.5*iqr #Lower range 
# onp <- subset(onp, onp$n_tokens_title > (Q[1] - 1.5*iqr) & onp$n_tokens_title < (Q[2]+1.5*iqr))
# 
# <!-------------------------- global_subjectivity ------------------------->
#   Q <- quantile(onp$global_subjectivity, probs =c(0.25, 0.75), na.rm = FALSE)
# iqr <- IQR(onp$global_subjectivity)
# up <- Q[2] + 1.5*iqr  #Upper Range
# low <- Q[1] - 1.5*iqr #Lower range 
# onp <- subset(onp, onp$global_subjectivity > (Q[1] - 1.5*iqr) & onp$global_subjectivity < (Q[2]+1.5*iqr))
# 
# <!-------------------------- kw_avg_avg ------------------------->
#   Q <- quantile(onp$kw_avg_avg, probs =c(0.25, 0.75), na.rm = FALSE)
# iqr <- IQR(onp$kw_avg_avg)
# up <- Q[2] + 1.5*iqr  #Upper Range
# low <- Q[1] - 1.5*iqr #Lower range 
# onp <- subset(onp, onp$kw_avg_avg > (Q[1] - 1.5*iqr) & onp$kw_avg_avg < (Q[2]+1.5*iqr))
# 
# <!-------------------------- kw_avg_max ------------------------->
#   Q <- quantile(onp$kw_avg_max, probs =c(0.25, 0.75), na.rm = FALSE)
# iqr <- IQR(onp$kw_avg_max)
# up <- Q[2] + 1.5*iqr  #Upper Range
# low <- Q[1] - 1.5*iqr #Lower range 
# onp <- subset(onp, onp$kw_avg_max > (Q[1] - 1.5*iqr) & onp$kw_avg_max < (Q[2]+1.5*iqr))
# 
# <!-------------------------- kw_max_avg ------------------------->
#   Q <- quantile(onp$kw_max_avg, probs =c(0.25, 0.75), na.rm = FALSE)
# iqr <- IQR(onp$kw_max_avg)
# up <- Q[2] + 1.5*iqr  #Upper Range
# low <- Q[1] - 1.5*iqr #Lower range 
# onp <- subset(onp, onp$kw_max_avg > (Q[1] - 1.5*iqr) & onp$kw_max_avg < (Q[2]+1.5*iqr))
# dim(onp)
# 
# <!-------------------------- kw_max_avg ------------------------->
#   Q <- quantile(onp$kw_max_avg, probs =c(0.25, 0.75), na.rm = FALSE)
# iqr <- IQR(onp$kw_max_avg)
# up <- Q[2] + 1.5*iqr  #Upper Range
# low <- Q[1] - 1.5*iqr #Lower range 
# onp <- subset(onp, onp$kw_max_avg > (Q[1] - 1.5*iqr) & onp$kw_max_avg < (Q[2]+1.5*iqr))
# dim(onp)
# 
# <!-------------------------- kw_min_max ------------------------->
#   Q <- quantile(onp$kw_min_max, probs =c(0.25, 0.75), na.rm = FALSE)
# iqr <- IQR(onp$kw_min_max)
# up <- Q[2] + 1.5*iqr  #Upper Range
# low <- Q[1] - 1.5*iqr #Lower range 
# onp <- subset(onp, onp$kw_min_max > (Q[1] - 1.5*iqr) & onp$kw_min_max < (Q[2]+1.5*iqr))
# dim(onp)
# 
# <!-------------------------- self_reference_avg_shares ------------------------->
#   Q <- quantile(onp$self_reference_avg_shares, probs =c(0.25, 0.75), na.rm = FALSE)
# iqr <- IQR(onp$self_reference_avg_shares)
# up <- Q[2] + 1.5*iqr  #Upper Range
# low <- Q[1] - 1.5*iqr #Lower range 
# onp <- subset(onp, onp$self_reference_avg_shares > (Q[1] - 1.5*iqr) & onp$self_reference_avg_shares < (Q[2]+1.5*iqr))
# dim(onp)
# 
# <!-------------------------- self_reference_max_shares ------------------------->
#   Q <- quantile(onp$self_reference_max_shares, probs =c(0.25, 0.75), na.rm = FALSE)
# iqr <- IQR(onp$self_reference_max_shares)
# up <- Q[2] + 1.5*iqr  #Upper Range
# low <- Q[1] - 1.5*iqr #Lower range 
# onp <- subset(onp, onp$self_reference_max_shares > (Q[1] - 1.5*iqr) & onp$self_reference_max_shares < (Q[2]+1.5*iqr))
# dim(onp)
# 
# <!-------------------------- self_reference_min_shares ------------------------->
#   Q <- quantile(onp$self_reference_min_shares, probs =c(0.25, 0.75), na.rm = FALSE)
# iqr <- IQR(onp$self_reference_min_shares)
# up <- Q[2] + 1.5*iqr  #Upper Range
# low <- Q[1] - 1.5*iqr #Lower range 
# onp <- subset(onp, onp$self_reference_min_shares > (Q[1] - 1.5*iqr) & onp$self_reference_min_shares < (Q[2]+1.5*iqr))
# dim(onp)

#From outlier detection we observe that if we start removing the outliers, the decrease in instances is huge
#It is turning out to be 0 instances after removal of outliers
#Even the target variable has huge number of outliers
#Considering the importance and loss of instances, let us go ahead with the outliers

# we have tried removing outliers, but if we remove outliers from the dataset, it is reducing the nrows to 7000 which is almost deleting 3/4 of the original data. so we went with 
# removing only the variables which are not relevent.

summary(d5$shares)
#The median is 1400, considering it, it is classified as Popular and unpopular

Feature <- d5
Feature <- as.data.frame(Feature)
str(d5)
Feature$Popularity <- ifelse(d1$shares > 1400, 1, 0)

#Scaling data is required since the values are out of range and comparison of the values is difficult

View(Feature)
dim(Feature)
str(Feature)
Feature <- subset(Feature, select = -shares)
Feature[,1:12]<-scale(Feature[,1:12])
Feature[,14:27] <- scale(Feature[,14:27])

# Removing self reference min shares, since we have max shares.
Feature <- subset(Feature, select = -c(self_reference_min_shares))
dim(Feature)

#########################################################

###################################################
#Split the dataset into training and test data - start
##################################################

set.seed(123)
i_split<- sample.split(row.names(Feature), 0.7) 
train<-Feature[i_split, ]
test<-Feature[!i_split, ]

dim(train)
dim(test)

# factorising is_weekend, popularity, channel to feed to the model.
train$Popularity <- as.factor(train$Popularity)
train$channel <- as.factor(train$channel)
train$is_weekend <- as.factor(train$is_weekend)

test$Popularity <- as.factor(test$Popularity)
test$channel <- as.factor(test$channel)
test$is_weekend <- as.factor(test$is_weekend)

str(train)
str(test)

# creating scale variable
str(train)
train[, 1:11] <- as.data.frame(scale(train[, 1:11]))
train[, 13:26] <- as.data.frame(scale(train[, 13:26]))

test[, 1:11] <- as.data.frame(scale(test[, 1:11]))
test[, 13:26] <- as.data.frame(scale(test[, 13:26]))

View(test)
View(train)

###################################################
#Split the dataset into training and test data - end
##################################################

#####################################################################################################################################################################################
                                                                              # Feature Selection - start
#####################################################################################################################################################################################

##### Variable Importance
# It is one of the classification models and builds Learning Vector Quantization model
# ensure results are repeatable
set.seed(123)
# prepare training scheme
control_function <- trainControl(method="repeatedcv", number=10, repeats=3)
# train the model
LVQmodel <- train(Popularity~., data=train, method="lvq", preProcess="scale", trControl=control_function)
# estimate variable importance
var_importance <- varImp(LVQmodel, scale=FALSE)
# summarize importance
print(var_importance)
# plot importance
plot(var_importance)

#*****************************************************************
#*Random Forest Feature selection technique using boruta package
#*****************************************************************

boruta_output <- Boruta(Popularity ~ ., data=na.omit(as.data.frame(train)), doTrace=0)
head(boruta_output)
boruta_signif <- names(boruta_output$finalDecision[boruta_output$finalDecision %in% c("Confirmed", "Tentative")])  
# collect Confirmed and Tentative variables
print(boruta_signif)
plot(boruta_output, cex.axis=1.0, las=2, xlab="Variables", main="Variable Importance")  # plot variable importance


#*****************************************************************
#* Recursive Feature elimination
#*****************************************************************

set.seed(123)
options(warn=-1)

dim(train)
subsets <- c(1:5, 10, 15, 18)

control <- rfeControl(functions = rfFuncs, # random forest
                      method = "repeatedcv", # repeated cv
                      repeats = 5, # number of repeats
                      number = 10) # number of folds

result_rfe1 <- rfe(x = train[,-18], 
                   y = train$Popularity, 
                   sizes = subsets,
                   rfeControl = control)

# Print the results
result_rfe1

# Print the selected features
predictors(result_rfe1)

# Print the results visually
ggplot(data = result_rfe1, metric = "Accuracy") + theme_bw()
ggplot(data = result_rfe1, metric = "Kappa") + theme_bw()

dim(train)
str(train)
train <- subset(train, select = c(self_reference_max_shares, global_subjectivity, global_sentiment_polarity, num_hrefs, is_weekend, global_rate_positive_words, num_imgs,
            title_sentiment_polarity, max_positive_polarity, n_tokens_title, avg_positive_polarity, kw_max_max, channel, title_subjectivity,
             num_keywords, num_videos, n_tokens_content, Popularity))

test <- subset(test, select = c(self_reference_max_shares, global_subjectivity, global_sentiment_polarity, num_hrefs, is_weekend, global_rate_positive_words, num_imgs,
                                  title_sentiment_polarity, max_positive_polarity, n_tokens_title, avg_positive_polarity, kw_max_max, title_subjectivity,
                                  num_keywords, num_videos, n_tokens_content, channel, Popularity))

######################################################################################################################################################################################
                                                                             # Feature Selection - end
######################################################################################################################################################################################

######################################################################################################################################################################################
                                                                             # Model Building - start
######################################################################################################################################################################################

###################################################################################### Naive Bayes ###################################################################################
x <- subset(train, select = -Popularity)
y <- train$Popularity

x$channel <- as.factor(x$channel)
x$is_weekend <- as.factor(x$is_weekend)

str(x)
# Training a model with training dataset
model = train(x,as.factor(y),'nb',trControl=trainControl(method='cv',number=10))
dim(test)
# Predicting a built model with test dataset
Predict <- predict(model,newdata = test)

# Predicting a built model with train dataset
Predict_train <- predict(model,newdata = train)

# Confusion Matrix
cm_train <- table(train$Popularity, Predict_train)
cm_test <- table(test$Popularity, Predict)

# Naive bayes is giving 59% accuracy with train dataset.
# Accuracy	Precision	Sensitivity	Specificity
# 0.594558559	0.677826087	0.341262313	0.841723033

# Naive bayes is giving 60% accuracy with test dataset.
# Accuracy	Precision	Sensitivity	Specificity
# 0.583116119	0.626052779	0.380806011	0.77936061

#Get the confusion matrix to see accuracy value and other parameter values > confusionMatrix(Predict, testing$Outcome )
dim(test)
dim(train)
p1 <- predict(model, train)
(tab1 <- table(p1, train$Popularity))

1 - sum(diag(tab1)) / sum(tab1)
0.4054414

p2 <- predict(model, test)
(tab2 <- table(p2, test$Popularity))

1 - sum(diag(tab2)) / sum(tab2)

# It seems that naive bayes is giving 42% misclassification which gives 58% accuracy.
0.4168839

######################################################################################## Decision tree  ###############################################################################
# The above reference link has the option to plot ROC curve.
train$Popularity <- as.factor(train$Popularity)
install.packages("rpart")
#Beautify tree
install.packages("rattle")
install.packages("rpart.plot")
install.packages("RColorBrewer")
install.packages("rattle")
rattle()
library(rattle)
library(rpart.plot)
library(RColorBrewer)
library(rpart)

decision_tree <- rpart(Popularity~., data = train, method="class", control = rpart.control(minsplit = 20, minbucket = 7, maxdepth = 10, usesurrogate = 2, xval =10 ))
decision_tree

#Plot tree
plot(decision_tree)
text(decision_tree)

#view1
prp(decision_tree, faclen = 0, cex = 0.8, extra = 1)

#view2 - total count at each node
total_count <- function(x, labs, digits, varlen)
{paste(labs, "\n\nn =", x$frame$n)}

prp(decision_tree, faclen = 0, cex = 0.8, node.fun=total_count)

#view3- fancy Plot
fancyRpartPlot(decision_tree)
printcp(decision_tree)
bestcp <- decision_tree$cptable[which.min(decision_tree$cptable[,"xerror"]),"CP"]

# we have taken bestcp froma bove code, lower the cp values it leads to creating a over-fitting model.
# Prune the tree using the best cp.
pruned <- prune(decision_tree, cp = bestcp)

# Plot pruned tree
prp(pruned, faclen = 0, cex = 0.8, extra = 1)

# confusion matrix (training data)
confusion.matrix <- table(train$Popularity, predict(pruned,type="class"))
rownames(confusion.matrix) <- paste("Actual", rownames(confusion.matrix), sep = ":")
colnames(confusion.matrix) <- paste("Pred", colnames(confusion.matrix), sep = ":")
print(confusion.matrix)

# #Accuracy	   Precision    	   Sensitivity	            Specificity
# 0.625009009	 0.612709259 	     0.654286757	            0.596440014

# From above results, decision tree is giving 62% accuracy and 65% sensitivity in training dataset

# confusion matrix (test data)
p <- predict(decision_tree, test, type = 'class')
confusion.matrix_test <- table(test$Popularity, p)
rownames(confusion.matrix_test) <- paste("Actual", rownames(confusion.matrix_test), sep = ":")
colnames(confusion.matrix_test) <- paste("Pred", colnames(confusion.matrix_test), sep = ":")
print(confusion.matrix_test)

# Accuracy and Sensitivity for test data
# Accuracy	Precision	Sensitivity	Specificity
# 0.621626167	0.607619048	0.653688525	0.590525095


#Scoring
install.packages("ROCR")
library(ROCR)
value1 = predict(pruned, train, type = "prob")
#Storing Model Performance Scores
pred_value <-prediction(value1[,2],train$Popularity)

# Calculating Area under Curve
perf_val <- performance(pred_value,"auc")
perf_val

# Plotting Lift curve
plot(performance(pred_value, measure="lift", x.measure="rpp"), colorize=TRUE)

# Our plot is having high maximum lift point which means, our model is better at predicting the result. where you can see it is going beyond 1.5 on y-axis.

# Calculating True Positive and False Positive Rate
perf_value <- performance(pred_value, "tpr", "fpr")

# Plot the ROC curve
plot(perf_value, col = "green", lwd = 1.5)

# AUC and ROC interpret
# When 0.5<AUC<1, there is a high chance that the classifier will be able to distinguish the positive class values from the negative class values. 
# This is so because the classifier is able to detect more numbers of True positives and True negatives than False negatives and False positives.

# our plot is having slight curve, which means it is is having AUC value between 0.5 & 1.

################################################################################### Random Forest ####################################################################################
# Installing package
install.packages("caTools")       # For sampling the dataset
install.packages("randomForest")  # For implementing random forest algorithm

# Loading package
library(caTools)
library(randomForest)

# Fitting Random Forest to the train dataset
set.seed(123)  # Setting seed
rf_x = subset(train, select = -Popularity)
rf_y = train$Popularity

classifier_RF = randomForest(rf_x,
                             rf_y,
                             ntree = 500)

classifier_RF
dim(rf_x)

# Result and Confusion matrix for training data prediction.
# all:
#   randomForest(x = rf_x, y = rf_y, ntree = 500) 
# Type of random forest: classification
# Number of trees: 500
# No. of variables tried at each split: 4
# 
# OOB estimate of  error rate: 35.4%
# Confusion matrix:
#   0    1 class.error
# 0 9143 4902    0.349021
# 1 4922 8783    0.359139

# Accuracy	Precision	Sensitivity	Specificity
# 0.645981982	0.641797589	0.640861	0.650978996

# Predicting the Test set results
y_pred = predict(classifier_RF, newdata = test)

# Confusion Matrix
confusion_mtx = table(test[,18], y_pred)
confusion_mtx

####### Result - start (Testing data) #########
# Accuracy	Precision	Sensitivity	Specificity
# 0.643487766	0.638783923	0.635075137	0.65164817


# No. of variables tried at each split: 4
# 
# OOB estimate of  error rate: 35.4%
# Confusion matrix:
#   0    1    
# 0 3934  2103
# 1 2137  3719

# we are getting accuracy of 64% and sensitivity of 64%
####### Result - end #########

# Plotting model
plot(classifier_RF)

# Importance plot
importance(classifier_RF)

# Variable importance plot
varImpPlot(classifier_RF)

################################################################################### XG Boost #########################################################################################
# We are using XG Boost rather than ADABoost. Because ADABoost is having a change of ending up with over-fitting model. which may perform poor in new dataset and ADABoost is slow
# compared to XGBoost. ADABoost is most suitable for low noise dataset.

# specifying the CV technique which will be passed into the train() function later and number parameter is the "k" in K-fold cross validation
train_control = trainControl(method = "cv", number = 5, search = "grid")

set.seed(123)

# Customsing the tuning grid
tuning_Grid <-  expand.grid(max_depth = c(3, 5, 7), 
                        nrounds = (1:10)*50,    # number of trees
                        # default values below
                        eta = 0.3,
                        gamma = 0,
                        subsample = 1,
                        min_child_weight = 1,
                        colsample_bytree = 0.6)


# training a XGboost Regression tree model while tuning hyper-parameters, in which we are setting few parameters in order to avoid over-fitting and leveraging the hyper parameters
# will help us to improve the accuracy.
options("install.lock"=FALSE)
install.packages('xgboost')     # for fitting the xgboost model

install.packages('caret')       # for general data preparation and model fitting

library(xgboost)
library(caret)
library(tidyverse)

# Model 
model_xgboost = train(Popularity~., data = train, method = "xgbTree", trControl = train_control, tuneGrid = tuning_Grid)

# summarising the results
print(model_xgboost)
# Compute feature importance matrix

dim(train)
#use model to make predictions on test data
pred_y_xgboost = predict(model_xgboost, test)
pred_x_xgboost = predict(model_xgboost, train)

# performance metrics on the test data
test_y_xgboost = test[,18]
test_x_xgboost = train[,18]

conf_mat = confusionMatrix(test_y_xgboost, pred_y_xgboost)
print(conf_mat)

conf_mat_train = confusionMatrix(test_x_xgboost, pred_x_xgboost)
print(conf_mat_train)

############ Result - start(Train) - XGBoost #########
# Confusion Matrix and Statistics
# 
# Reference
# Prediction   
#    0    1
# 0 9440 4605
# 1 4796 8909
# 

# Accuracy  	Precision	  Sensitivity	Specificity
# 0.661225225	0.659242267	0.650054725	0.672125311

# Accuracy : 0.6612          
# 95% CI : (0.6556, 0.6668)
# No Information Rate : 0.513           
# P-Value [Acc > NIR] : < 2e-16         
# 
# Kappa : 0.3222          
# 
# Mcnemar's Test P-Value : 0.05004         
#                                           
#             Sensitivity : 0.6631          
#             Specificity : 0.6592          
#          Pos Pred Value : 0.6721          
#          Neg Pred Value : 0.6501          
#              Prevalence : 0.5130          
#          Detection Rate : 0.3402          
#    Detection Prevalence : 0.5061          
#       Balanced Accuracy : 0.6612          
#                                           
#        'Positive' Class : 0 
# ########### Result - end(Train) #########

# ########### Result - start(Test) - XGBoost #########
# Reference
# Prediction    0    1
# 0 4063 1974
# 1 2167 3689

# Accuracy  	Precision	     Sensitivity	Specificity
# 0.65181199	0.651421508	   0.629952186	0.673016399

# Accuracy : 0.6518          
# 95% CI : (0.6432, 0.6604)
# No Information Rate : 0.5238          
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.3031          
# 
# Mcnemar's Test P-Value : 0.002848        
#                                           
#             Sensitivity : 0.6522          
#             Specificity : 0.6514          
#          Pos Pred Value : 0.6730          
#          Neg Pred Value : 0.6300          
#              Prevalence : 0.5238          
#          Detection Rate : 0.3416          
#    Detection Prevalence : 0.5076          
#       Balanced Accuracy : 0.6518          
#                                           
#        'Positive' Class : 0   
########### Result - end(Test) ############

######################################################################################################################################################################################
                                                                                     # Model Building - end
######################################################################################################################################################################################

# Plotting graph of important variables in XGBoost algorithm
#define predictor and response variables in training set
train_x <- data.matrix(train[, -18])
train_y <- train[,18]

#define predictor and response variables in testing set
test_x <- data.matrix(test[, -18])
test_y <- test[, 18]

#define final training and testing sets
xgb_train <- xgb.DMatrix(data = train_x, label = train_y)
xgb_test <- xgb.DMatrix(data = test_x, label = test_y)

#defining a watchlist
watchlist <- list(train=xgb_train, test=xgb_test)

#fit XGBoost model and display training and testing data at each iteartion
model <- xgb.train(data = xgb_train, max.depth = 3, watchlist=watchlist, nrounds = 100)

#define final model
model_xgboost <- xgboost(data = xgb_train, max.depth = 3, nrounds = 86, verbose = 0)
summary(model_xgboost)

library(caret) 			# for general data preparation and model fitting

library(rpart.plot)

library(tidyverse)

# Compute feature importance matrix
importance_matrix <- xgb.importance(colnames(xgb_train), model = model_xgboost)
importance_matrix

xgb.plot.importance(importance_matrix[1:17,])



















































