<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
body {font-family: Arial, Helvetica, sans-serif;}
form {border: 3px solid #f1f1f1;}

button {
  background-color: #4CAF50;
  color: white;
  padding: 14px 20px;
  margin: 8px 0;
  border: none;
  cursor: pointer;
  width: 100%;
}

button:hover {
  opacity: 0.8;
}

.cancelbtn {
  width: auto;
  padding: 10px 18px;
  background-color: #f44336;
}

.imgcontainer {
  text-align: center;
  margin: 24px 0 12px 0;
}

img.avatar {
  width: 40%;
  border-radius: 50%;
}

.container {
  padding: 16px;
}

span.psw {
  float: right;
  padding-top: 16px;
}

/* Change styles for span and cancel button on extra small screens */
@media screen and (max-width: 300px) {
  span.psw {
     display: block;
     float: none;
  }
  .cancelbtn {
    width: 100%;
  }
} 

button{
  color: #1a1a1a;
}

form{
  width: 100%;
}
.content2 {
  /* background: #76b882; */
  float: left;
  padding: 20px;
  width: 500px;
  margin: auto;
    
}
.content3{
  /* background: #76b882; */
  float: right;
  padding: 20px;
  /* width: 360px; */
  margin: auto;
}

body {
  background: #76b852;
  background: -webkit-linear-gradient(right, #76b852, #8DC26F);
  background: -moz-linear-gradient(right, #76b852, #8DC26F);
  background: -o-linear-gradient(right, #76b852, #8DC26F);
  background: linear-gradient(to left, #76b852, #8DC26F);
  font-family: "Roboto", sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;      
}

</style>

<script src="http://ajax.googleapis.com/ajax/libs/jquery/1.9.1/jquery.js"></script>
</head>


<body>

<?php 
  $message = "";

  if(isset($_POST['training'])){
    $message = "Loss: \n";
    $message .= shell_exec('/opt/anaconda3/bin/python3.7 ./python/Training.py 2>&1');
    $message .= "\n";
    // $message= "The training function is called.";
    // $output = shell_exec('/opt/anaconda3/bin/python3.7 ./python/Training.py 2>&1');
    // echo $output . "<br>"; // . $command;
  }
  if(isset($_POST['testing'])){
    $message .= "\nOutput: \n";
    $message = shell_exec('/opt/anaconda3/bin/python3.7 ./python/Testing.py 2>&1');
    $message .= "\n";
    // $message="The testing function is called.";
    // $output = shell_exec('/opt/anaconda3/bin/python3.7 ./python/Testing.py 2>&1');
    // echo $output . "<br>";// . $command;
  }
?>


<div class="content2">

  <h1>Proxy Signature Detection</h1>


  <form method="post">
  <!-- <input type="text" name="txt" value="<?php if(isset($message)){ echo $message;}?>" > -->
    <button value="training" type="submit" name="training"><b>Training</b></button>

  </form>

  <form method="post">
    <button value="testing" type="submit" name="testing"><b>Testing</b></button>
  </form>

</div>


<div class="content3">
<br><br>


<!--------------------------------------
printing output.
-->

<h2>OUTPUT</h2>
  <textarea rows="30" cols="50" name="output" ><?php if(isset($message)){ echo $message;}?></textarea>
  <!-- <textarea rows="10" cols="50"></textarea> -->
<!--------------------------------------------->

</div>

</body>
</html>

