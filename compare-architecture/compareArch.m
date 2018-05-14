arch1 = dlmread('x_y1_y2_arch1','\t');
arch1_x = arch1(:,1);
arch1_y1 = arch1(:,2);
arch1_y2 = arch1(:,3);
arch2 = dlmread('x_y1_y2_arch2','\t');
arch2_x = arch2(:,1);
arch2_y1 = arch2(:,2);
arch2_y2 = arch2(:,3);
arch3 = dlmread('x_y1_y2_arch3','\t');
arch3_x = arch3(:,1);
arch3_y1 = arch3(:,2);
arch3_y2 = arch3(:,3);
arch4 = dlmread('x_y1_y2_arch4','\t');
arch4_x = arch4(:,1);
arch4_y1 = arch4(:,2);
arch4_y2 = arch4(:,3);

figure,hold
plot(arch1_x(arch1_y1>0),arch1_y1(arch1_y1>0),'o-','Color','r','MarkerSize',5,...
    'MarkerEdgeColor','red');
plot(arch2_x(arch2_y1>0),arch2_y1(arch2_y1>0),'o-','Color','g','MarkerSize',5,...
    'MarkerEdgeColor','green');
plot(arch3_x(arch3_y1>0),arch3_y1(arch3_y1>0),'o-','Color','b','MarkerSize',5,...
    'MarkerEdgeColor','blue');
plot(arch4_x(arch4_y1>0),arch4_y1(arch4_y1>0),'o-','Color','m','MarkerSize',5,...
    'MarkerEdgeColor','m');
xlabel('Iterations')
ylabel('Cross-entropy Loss')
legend('arch1-train','arch2-train','arch3-train','arch4-train')


figure,hold
plot(arch1_x(arch1_y2>0),arch1_y2(arch1_y2>0),'o-','Color','r','MarkerSize',5,...
    'MarkerEdgeColor','red','MarkerFaceColor',[1 .6 .6]);
plot(arch2_x(arch2_y2>0),arch2_y2(arch2_y2>0),'o-','Color','g','MarkerSize',5,...
    'MarkerEdgeColor','green','MarkerFaceColor',[.6 1 .6]);
plot(arch3_x(arch3_y2>0),arch3_y2(arch3_y2>0),'o-','Color','b','MarkerSize',5,...
    'MarkerEdgeColor','blue','MarkerFaceColor',[.6 .6 1]);
plot(arch4_x(arch4_y2>0),arch4_y2(arch4_y2>0),'o-','Color','m','MarkerSize',5,...
    'MarkerEdgeColor','m','MarkerFaceColor','m');
xlabel('Iterations')
ylabel('Cross-entropy Loss')
legend('arch1-test','arch2-test','arch3-test','arch4-test')