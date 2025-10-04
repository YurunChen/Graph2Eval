USE wordpress;

-- Insert sample posts
INSERT INTO wp_posts (post_author, post_date, post_date_gmt, post_content, post_title, post_excerpt, post_status, comment_status, ping_status, post_password, post_name, to_ping, pinged, post_modified, post_modified_gmt, post_content_filtered, post_parent, guid, menu_order, post_type, post_mime_type, comment_count) VALUES
(1, NOW(), NOW(), 'Python is a high-level programming language known for its simplicity and readability.', 'Introduction to Python Programming', 'Learn the basics of Python programming language', 'publish', 'open', 'open', '', 'python-programming', '', '', NOW(), NOW(), '', 0, 'http://localhost:8081/?p=1', 0, 'post', '', 0),
(1, NOW(), NOW(), 'Artificial Intelligence is rapidly transforming our world.', 'The Future of AI', 'Exploring the current state and future of AI technology', 'publish', 'open', 'open', '', 'future-of-ai', '', '', NOW(), NOW(), '', 0, 'http://localhost:8081/?p=2', 0, 'post', '', 0),
(1, NOW(), NOW(), 'Maintaining a healthy diet is essential for overall well-being.', 'Healthy Eating Habits', 'Practical tips for developing healthy eating habits', 'publish', 'open', 'open', '', 'healthy-eating', '', '', NOW(), NOW(), '', 0, 'http://localhost:8081/?p=3', 0, 'post', '', 0),
(1, NOW(), NOW(), 'Welcome to our website! We provide valuable information on various topics.', 'About Us', 'Learn more about our mission and values', 'publish', 'open', 'open', '', 'about-us', '', '', NOW(), NOW(), '', 0, 'http://localhost:8081/?p=4', 0, 'page', '', 0),
(1, NOW(), NOW(), 'Contact us for any questions or feedback.', 'Contact Us', 'Get in touch with us', 'publish', 'open', 'open', '', 'contact-us', '', '', NOW(), NOW(), '', 0, 'http://localhost:8081/?p=5', 0, 'page', '', 0);

-- Insert sample comments
INSERT INTO wp_comments (comment_post_ID, comment_author, comment_author_email, comment_date, comment_date_gmt, comment_content, comment_approved, comment_agent, comment_type) VALUES
(1, 'John Doe', 'john@example.com', NOW(), NOW(), 'Great article! Python is fantastic for beginners.', '1', 'Mozilla/5.0', 'comment'),
(2, 'Jane Smith', 'jane@example.com', NOW(), NOW(), 'AI is fascinating but also a bit scary.', '1', 'Mozilla/5.0', 'comment'),
(3, 'Bob Wilson', 'bob@example.com', NOW(), NOW(), 'These tips are really practical.', '1', 'Mozilla/5.0', 'comment');

-- Update comment counts
UPDATE wp_posts SET comment_count = 1 WHERE ID = 1;
UPDATE wp_posts SET comment_count = 1 WHERE ID = 2;
UPDATE wp_posts SET comment_count = 1 WHERE ID = 3;

-- Set basic options (ignore if already exist)
INSERT IGNORE INTO wp_options (option_name, option_value, autoload) VALUES
('blogname', 'My Test Site', 'yes'),
('blogdescription', 'A comprehensive website with articles', 'yes'),
('posts_per_page', '10', 'yes');

SELECT 'WordPress Demo Data Insertion Complete!' as status;
