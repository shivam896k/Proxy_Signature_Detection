<?php

    $output = shell_exec('/opt/anaconda3/bin/python3.7 ./python/Testing.py 2>&1');
    echo $output . "<br>";// . $command;

?>
