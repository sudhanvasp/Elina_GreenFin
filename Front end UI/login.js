document.getElementById('login-form').addEventListener('submit', function(event) {
    event.preventDefault();
    var username = document.getElementById('username').value;
    var password = document.getElementById('password').value;
  
    // Check if the credentials are as expected
    if(username === 'admin' && password === 'password') {
      // Logic for a successful login
      alert('Login successful!');
      // Redirect to another page or dashboard
      window.location.href = 'index.html';
    } else {
      // Logic for a failed login
      alert('Incorrect username or password!');
    }
  });
  