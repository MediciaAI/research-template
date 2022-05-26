library(dplyr)
library(ggplot2)
pdf(NULL) # To stop creating the Rplots.pdf file.
ggplot(data=iris, aes(x=Sepal.Length, y=Sepal.Width, color=Species)) + geom_point(size=3)
ggsave("r_plot.png")
