-- WordPress Demo Data Insertion Script
-- Inserts sample posts, pages, categories, and comments

USE wordpress;

-- Insert sample categories
INSERT INTO wp_terms (name, slug, term_group) VALUES
('Technology', 'technology', 0),
('Science', 'science', 0),
('Business', 'business', 0),
('Health', 'health', 0),
('Travel', 'travel', 0);

-- Insert term taxonomy for categories
INSERT INTO wp_term_taxonomy (term_id, taxonomy, description, parent, count) VALUES
(1, 'category', 'Technology related articles', 0, 0),
(2, 'category', 'Scientific discoveries and research', 0, 0),
(3, 'category', 'Business and entrepreneurship', 0, 0),
(4, 'category', 'Health and wellness', 0, 0),
(5, 'category', 'Travel guides and tips', 0, 0);

-- Insert sample posts
INSERT INTO wp_posts (post_author, post_date, post_date_gmt, post_content, post_title, post_excerpt, post_status, comment_status, ping_status, post_password, post_name, to_ping, pinged, post_modified, post_modified_gmt, post_content_filtered, post_parent, guid, menu_order, post_type, post_mime_type, comment_count) VALUES
(1, NOW(), NOW(), 'Python is a high-level, interpreted programming language known for its simplicity and readability. It was created by Guido van Rossum and first released in 1991. Python supports multiple programming paradigms, including procedural, object-oriented, and functional programming.', 'Introduction to Python Programming', 'Learn the basics of Python programming language', 'publish', 'open', 'open', '', 'introduction-to-python-programming', '', '', NOW(), NOW(), '', 0, 'http://localhost:8081/?p=1', 0, 'post', '', 0),
(1, NOW(), NOW(), 'Artificial Intelligence (AI) is rapidly transforming our world. From self-driving cars to virtual assistants, AI is becoming an integral part of our daily lives. This article explores the current state of AI and its potential future developments.', 'The Future of Artificial Intelligence', 'Exploring the current state and future of AI technology', 'publish', 'open', 'open', '', 'the-future-of-artificial-intelligence', '', '', NOW(), NOW(), '', 0, 'http://localhost:8081/?p=2', 0, 'post', '', 0),
(1, NOW(), NOW(), 'Maintaining a healthy diet is essential for overall well-being. This article provides practical tips for developing healthy eating habits, including meal planning, portion control, and choosing nutritious foods.', 'Healthy Eating Habits', 'Practical tips for developing healthy eating habits', 'publish', 'open', 'open', '', 'healthy-eating-habits', '', '', NOW(), NOW(), '', 0, 'http://localhost:8081/?p=3', 0, 'post', '', 0),
(1, NOW(), NOW(), 'Entrepreneurship can be both challenging and rewarding. This guide covers the essential steps for starting your own business, from idea generation to market research and funding options.', 'Starting Your Own Business', 'Essential guide for entrepreneurs', 'publish', 'open', 'open', '', 'starting-your-own-business', '', '', NOW(), NOW(), '', 0, 'http://localhost:8081/?p=4', 0, 'post', '', 0),
(1, NOW(), NOW(), 'Tokyo is a fascinating city that combines traditional Japanese culture with modern technology. This travel guide covers the best places to visit, local cuisine, transportation, and cultural experiences.', 'Travel Guide: Tokyo, Japan', 'Comprehensive guide to exploring Tokyo', 'publish', 'open', 'open', '', 'travel-guide-tokyo-japan', '', '', NOW(), NOW(), '', 0, 'http://localhost:8081/?p=5', 0, 'post', '', 0);

-- Link posts to categories
INSERT INTO wp_term_relationships (object_id, term_taxonomy_id, term_order) VALUES
(1, 1, 0), -- Python Programming -> Technology
(2, 1, 0), -- AI -> Technology
(3, 4, 0), -- Healthy Eating -> Health
(4, 3, 0), -- Business -> Business
(5, 5, 0); -- Tokyo Travel -> Travel

-- Insert sample pages
INSERT INTO wp_posts (post_author, post_date, post_date_gmt, post_content, post_title, post_excerpt, post_status, comment_status, ping_status, post_password, post_name, to_ping, pinged, post_modified, post_modified_gmt, post_content_filtered, post_parent, guid, menu_order, post_type, post_mime_type, comment_count) VALUES
(1, NOW(), NOW(), 'Welcome to our website! We are dedicated to providing valuable information and insights on various topics including technology, science, business, health, and more.', 'About Us', 'Learn more about our mission and values', 'publish', 'open', 'open', '', 'about-us', '', '', NOW(), NOW(), '', 0, 'http://localhost:8081/?page_id=6', 0, 'page', '', 0),
(1, NOW(), NOW(), 'Thank you for visiting our website! We appreciate your interest in our content. If you have any questions, suggestions, or feedback, please don''t hesitate to reach out to us.', 'Contact Us', 'Get in touch with us', 'publish', 'open', 'open', '', 'contact-us', '', '', NOW(), NOW(), '', 0, 'http://localhost:8081/?page_id=7', 0, 'page', '', 0);

-- Insert sample comments
INSERT INTO wp_comments (comment_post_ID, comment_author, comment_author_email, comment_author_url, comment_author_IP, comment_date, comment_date_gmt, comment_content, comment_karma, comment_approved, comment_agent, comment_type, comment_parent, user_id) VALUES
(1, 'John Doe', 'john@example.com', '', '127.0.0.1', NOW(), NOW(), 'Great article! Python is indeed a fantastic language for beginners.', 0, '1', 'Mozilla/5.0', 'comment', 0, 0),
(2, 'Jane Smith', 'jane@example.com', '', '127.0.0.1', NOW(), NOW(), 'AI is fascinating but also a bit scary. What do you think about job displacement?', 0, '1', 'Mozilla/5.0', 'comment', 0, 0),
(3, 'Bob Wilson', 'bob@example.com', '', '127.0.0.1', NOW(), NOW(), 'These tips are really practical. I''ve been trying to eat healthier lately.', 0, '1', 'Mozilla/5.0', 'comment', 0, 0);

-- Update comment counts for posts
UPDATE wp_posts SET comment_count = 1 WHERE ID = 1;
UPDATE wp_posts SET comment_count = 1 WHERE ID = 2;
UPDATE wp_posts SET comment_count = 1 WHERE ID = 3;

-- Update category counts
UPDATE wp_term_taxonomy SET count = 2 WHERE term_id = 1; -- Technology: 2 posts
UPDATE wp_term_taxonomy SET count = 1 WHERE term_id = 3; -- Business: 1 post
UPDATE wp_term_taxonomy SET count = 1 WHERE term_id = 4; -- Health: 1 post
UPDATE wp_term_taxonomy SET count = 1 WHERE term_id = 5; -- Travel: 1 post

-- Set up theme options
INSERT INTO wp_options (option_name, option_value, autoload) VALUES
('blogname', 'My Test Site', 'yes'),
('blogdescription', 'A comprehensive website with articles on technology, science, business, health, and more', 'yes'),
('users_can_register', '0', 'yes'),
('posts_per_page', '10', 'yes'),
('permalink_structure', '/%postname%/', 'yes');

-- Update post GUIDs to use proper format
UPDATE wp_posts SET guid = CONCAT('http://localhost:8081/?p=', ID) WHERE post_type IN ('post', 'page');

-- Set up user capabilities
INSERT INTO wp_usermeta (user_id, meta_key, meta_value) VALUES
(1, 'wp_capabilities', 'a:1:{s:13:"administrator";b:1;}'),
(1, 'wp_user_level', '10');

-- Display summary
SELECT 'WordPress Demo Data Insertion Complete!' as status;
SELECT COUNT(*) as total_posts FROM wp_posts WHERE post_type = 'post' AND post_status = 'publish';
SELECT COUNT(*) as total_pages FROM wp_posts WHERE post_type = 'page' AND post_status = 'publish';
SELECT COUNT(*) as total_comments FROM wp_comments WHERE comment_approved = '1';
SELECT COUNT(*) as total_categories FROM wp_term_taxonomy WHERE taxonomy = 'category';


