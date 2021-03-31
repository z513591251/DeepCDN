use strict;
use warnings;

while (<>) {
   chomp;
   for(my $num=0; $num<10; $num++) {
       my $name=$num+1;
      if ($.%10==$num) {
      open (A,">>test$name.txt") or die;
        print A "$_\n";
       } else {
        open (B,">>train$name.txt") or die;
      print B "$_\n";
      }
  }
}

          