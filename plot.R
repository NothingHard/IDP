library(data.table)
library(ggplot2)

res_file <- grep("result",list.files("output/",full.names = T),value = T)
log_file <- grep("log",list.files("output/",full.names = T),value=T)

for(i in seq_along(res_file)){
    if(i==1){
        f <- fread(res_file[i])
    }
    else{
        tf <- fread(res_file[i])
        f <- rbind(f,tf)
    }
}
rm(tf)
f$IDP <- f$IDP*100
f$accu <- f$accu*100
ggplot(f,aes(x=IDP,y=accu,colour=profile))+
    scale_y_continuous(breaks = seq(50,100,by=5))+
    scale_x_continuous(breaks = seq(10,100,by=10))+
    geom_line(size=1.25)+
    geom_point(size=2)+
    ggtitle("MLP (MNIST)")+
    xlab("IDP (%)")+
    ylab("Classification accuracy (%)")+
    theme_bw()+
    coord_cartesian(ylim = c(50, 100)) +
    theme(plot.title = element_text(hjust=0.5))
ggsave("output/result.png")

#### plot log ####
for(i in seq_along(log_file)){
    f <- fread(log_file[i])
    f$epoch <- seq(1,nrow(f))
    pf <- melt(f[,.(epoch,accu,val_accu)],
               id.vars = "epoch",
               variable.name = "type")
    ggplot(pf,aes(x=epoch,y=value,colour=type))+
        geom_line()+
        geom_point()+
        ggtitle("Training log (accuracy)")+
        xlab("Epoch")+
        ylab("Accuracy")+
        theme_bw()+
        theme(plot.title = element_text(hjust=0.5))
    ggsave(gsub("\\.csv","_accu\\.png",log_file[i]))
    
    pf <- melt(f[,.(epoch,loss,val_loss)],
               id.vars = "epoch",
               variable.name = "type")
    ggplot(pf,aes(x=epoch,y=value,colour=type))+
        geom_line()+
        ggtitle("Training log (loss)")+
        xlab("Epoch")+
        ylab("Loss (CE)")+
        theme_bw()+
        theme(plot.title = element_text(hjust=0.5))
    ggsave(gsub("\\.csv","_loss\\.png",log_file[i]))
}
