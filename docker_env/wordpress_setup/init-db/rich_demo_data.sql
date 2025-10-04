USE wordpress;

-- Insert more sample categories
INSERT IGNORE INTO wp_terms (name, slug, term_group) VALUES
('Technology', 'technology', 0),
('Science', 'science', 0),
('Business', 'business', 0),
('Health', 'health', 0),
('Travel', 'travel', 0),
('Food', 'food', 0),
('Sports', 'sports', 0),
('Entertainment', 'entertainment', 0),
('Education', 'education', 0),
('Lifestyle', 'lifestyle', 0);

-- Insert term taxonomy for categories
INSERT IGNORE INTO wp_term_taxonomy (term_id, taxonomy, description, parent, count) VALUES
(1, 'category', 'Technology related articles', 0, 0),
(2, 'category', 'Scientific discoveries and research', 0, 0),
(3, 'category', 'Business and entrepreneurship', 0, 0),
(4, 'category', 'Health and wellness', 0, 0),
(5, 'category', 'Travel guides and tips', 0, 0),
(6, 'category', 'Food and recipes', 0, 0),
(7, 'category', 'Sports and fitness', 0, 0),
(8, 'category', 'Movies, music, and entertainment', 0, 0),
(9, 'category', 'Learning and education', 0, 0),
(10, 'category', 'Lifestyle and personal development', 0, 0);

-- Insert more sample posts
INSERT IGNORE INTO wp_posts (ID, post_author, post_date, post_date_gmt, post_content, post_title, post_excerpt, post_status, comment_status, ping_status, post_password, post_name, to_ping, pinged, post_modified, post_modified_gmt, post_content_filtered, post_parent, guid, menu_order, post_type, post_mime_type, comment_count) VALUES
(10, 1, NOW(), NOW(), 'Machine Learning is revolutionizing industries across the globe. From healthcare to finance, ML algorithms are transforming how we process data and make decisions. This comprehensive guide covers the fundamentals of machine learning, including supervised learning, unsupervised learning, and deep learning techniques.', 'Machine Learning Fundamentals', 'Complete guide to machine learning concepts and applications', 'publish', 'open', 'open', '', 'machine-learning-fundamentals', '', '', NOW(), NOW(), '', 0, 'http://localhost:8081/?p=10', 0, 'post', '', 0),

(11, 1, NOW(), NOW(), 'React has become the go-to framework for building modern web applications. With its component-based architecture and powerful state management, React enables developers to create scalable and maintainable user interfaces. Learn the core concepts including JSX, components, hooks, and state management.', 'Building Modern Web Apps with React', 'Learn React development from basics to advanced concepts', 'publish', 'open', 'open', '', 'building-modern-web-apps-react', '', '', NOW(), NOW(), '', 0, 'http://localhost:8081/?p=11', 0, 'post', '', 0),

(12, 1, NOW(), NOW(), 'Climate change represents one of the greatest challenges of our time. Rising global temperatures, extreme weather events, and environmental degradation require immediate action. This article explores sustainable solutions, renewable energy alternatives, and individual actions we can take to combat climate change.', 'Climate Change and Sustainable Solutions', 'Exploring environmental challenges and sustainable solutions', 'publish', 'open', 'open', '', 'climate-change-sustainable-solutions', '', '', NOW(), NOW(), '', 0, 'http://localhost:8081/?p=12', 0, 'post', '', 0),

(13, 1, NOW(), NOW(), 'Digital marketing has transformed how businesses connect with customers. From social media advertising to search engine optimization, digital strategies offer unprecedented reach and targeting capabilities. Learn about content marketing, email campaigns, social media strategies, and analytics tools.', 'Digital Marketing Strategies for 2024', 'Comprehensive guide to modern digital marketing', 'publish', 'open', 'open', '', 'digital-marketing-strategies-2024', '', '', NOW(), NOW(), '', 0, 'http://localhost:8081/?p=13', 0, 'post', '', 0),

(14, 1, NOW(), NOW(), 'Mental health awareness has never been more important. With increasing stress levels and social pressures, understanding mental wellness is crucial for personal and professional success. This article covers stress management techniques, mindfulness practices, and when to seek professional help.', 'Mental Health and Wellness Guide', 'Complete guide to maintaining mental health and wellness', 'publish', 'open', 'open', '', 'mental-health-wellness-guide', '', '', NOW(), NOW(), '', 0, 'http://localhost:8081/?p=14', 0, 'post', '', 0),

(15, 1, NOW(), NOW(), 'Mediterranean cuisine offers a perfect blend of flavor, nutrition, and cultural heritage. Known for its emphasis on fresh ingredients, olive oil, and seafood, the Mediterranean diet has been linked to numerous health benefits. Discover classic recipes, cooking techniques, and nutritional insights.', 'Mediterranean Cuisine and Healthy Cooking', 'Explore the flavors and health benefits of Mediterranean cooking', 'publish', 'open', 'open', '', 'mediterranean-cuisine-healthy-cooking', '', '', NOW(), NOW(), '', 0, 'http://localhost:8081/?p=15', 0, 'post', '', 0),

(16, 1, NOW(), NOW(), 'Fitness and exercise are fundamental to maintaining good health and quality of life. Whether you are a beginner or an experienced athlete, having a structured workout routine can help you achieve your goals. This guide covers strength training, cardio exercises, and recovery techniques.', 'Complete Fitness and Exercise Guide', 'Everything you need to know about fitness and exercise', 'publish', 'open', 'open', '', 'complete-fitness-exercise-guide', '', '', NOW(), NOW(), '', 0, 'http://localhost:8081/?p=16', 0, 'post', '', 0),

(17, 1, NOW(), NOW(), 'The film industry continues to evolve with new technologies and storytelling techniques. From blockbuster franchises to independent films, cinema offers diverse experiences for audiences worldwide. Explore film analysis, industry trends, and recommendations for must-watch movies.', 'Cinema and Film Industry Trends', 'Exploring the world of cinema and entertainment', 'publish', 'open', 'open', '', 'cinema-film-industry-trends', '', '', NOW(), NOW(), '', 0, 'http://localhost:8081/?p=17', 0, 'post', '', 0),

(18, 1, NOW(), NOW(), 'Online education has revolutionized learning accessibility and flexibility. With the rise of e-learning platforms, students can access quality education from anywhere in the world. This article examines online learning strategies, digital tools, and the future of education technology.', 'Online Education and E-Learning', 'The future of digital learning and education technology', 'publish', 'open', 'open', '', 'online-education-elearning', '', '', NOW(), NOW(), '', 0, 'http://localhost:8081/?p=18', 0, 'post', '', 0),

(19, 1, NOW(), NOW(), 'Personal development is a lifelong journey of self-improvement and growth. Setting goals, developing new skills, and maintaining work-life balance are essential components of a fulfilling life. Learn practical strategies for personal growth, productivity, and achieving your aspirations.', 'Personal Development and Life Skills', 'Strategies for personal growth and self-improvement', 'publish', 'open', 'open', '', 'personal-development-life-skills', '', '', NOW(), NOW(), '', 0, 'http://localhost:8081/?p=19', 0, 'post', '', 0);

-- Insert more sample pages
INSERT IGNORE INTO wp_posts (ID, post_author, post_date, post_date_gmt, post_content, post_title, post_excerpt, post_status, comment_status, ping_status, post_password, post_name, to_ping, pinged, post_modified, post_modified_gmt, post_content_filtered, post_parent, guid, menu_order, post_type, post_mime_type, comment_count) VALUES
(20, 1, NOW(), NOW(), 'Welcome to our comprehensive knowledge platform! We are dedicated to providing high-quality, well-researched content across multiple disciplines including technology, science, business, health, and lifestyle. Our team of experts works tirelessly to bring you the latest insights and practical information to help you stay informed and make better decisions in your personal and professional life.', 'About Our Platform', 'Learn about our mission and expertise', 'publish', 'open', 'open', '', 'about-platform', '', '', NOW(), NOW(), '', 0, 'http://localhost:8081/?p=20', 0, 'page', '', 0),

(21, 1, NOW(), NOW(), 'We love hearing from our readers! Whether you have questions, suggestions, feedback, or collaboration ideas, please do not hesitate to reach out. You can contact us via email at info@example.com, follow us on social media, or use the contact form below. We typically respond within 24-48 hours.', 'Contact Information', 'Get in touch with our team', 'publish', 'open', 'open', '', 'contact-information', '', '', NOW(), NOW(), '', 0, 'http://localhost:8081/?p=21', 0, 'page', '', 0),

(22, 1, NOW(), NOW(), 'Our privacy policy outlines how we collect, use, and protect your personal information. We are committed to maintaining the privacy and security of our users data. This policy covers cookie usage, data collection practices, third-party services, and your rights regarding personal information.', 'Privacy Policy', 'Our commitment to protecting your privacy', 'publish', 'closed', 'closed', '', 'privacy-policy', '', '', NOW(), NOW(), '', 0, 'http://localhost:8081/?p=22', 0, 'page', '', 0),

(23, 1, NOW(), NOW(), 'These terms of service govern your use of our website and services. By accessing our content, you agree to these terms and conditions. Please read them carefully before using our platform. These terms cover user responsibilities, content usage rights, and limitation of liability.', 'Terms of Service', 'Terms and conditions for using our platform', 'publish', 'closed', 'closed', '', 'terms-of-service', '', '', NOW(), NOW(), '', 0, 'http://localhost:8081/?p=23', 0, 'page', '', 0);

-- Link posts to categories
INSERT IGNORE INTO wp_term_relationships (object_id, term_taxonomy_id, term_order) VALUES
(10, 1, 0), -- Machine Learning -> Technology
(11, 1, 0), -- React -> Technology
(12, 2, 0), -- Climate Change -> Science
(13, 3, 0), -- Digital Marketing -> Business
(14, 4, 0), -- Mental Health -> Health
(15, 6, 0), -- Mediterranean Cuisine -> Food
(16, 7, 0), -- Fitness -> Sports
(17, 8, 0), -- Cinema -> Entertainment
(18, 9, 0), -- Online Education -> Education
(19, 10, 0); -- Personal Development -> Lifestyle

-- Insert sample tags
INSERT IGNORE INTO wp_terms (name, slug, term_group) VALUES
('Programming', 'programming', 0),
('Web Development', 'web-development', 0),
('Data Science', 'data-science', 0),
('Startup', 'startup', 0),
('Nutrition', 'nutrition', 0),
('Mindfulness', 'mindfulness', 0),
('Sustainability', 'sustainability', 0),
('Digital Marketing', 'digital-marketing', 0),
('JavaScript', 'javascript', 0),
('Python', 'python', 0),
('Fitness', 'fitness', 0),
('Movies', 'movies', 0),
('Online Learning', 'online-learning', 0),
('Self Improvement', 'self-improvement', 0),
('Cooking', 'cooking', 0);

-- Insert term taxonomy for tags
INSERT IGNORE INTO wp_term_taxonomy (term_id, taxonomy, description, parent, count) VALUES
(11, 'post_tag', 'Programming related content', 0, 0),
(12, 'post_tag', 'Web development topics', 0, 0),
(13, 'post_tag', 'Data science and analytics', 0, 0),
(14, 'post_tag', 'Startup and entrepreneurship', 0, 0),
(15, 'post_tag', 'Nutrition and diet', 0, 0),
(16, 'post_tag', 'Mindfulness and meditation', 0, 0),
(17, 'post_tag', 'Environmental sustainability', 0, 0),
(18, 'post_tag', 'Digital marketing strategies', 0, 0),
(19, 'post_tag', 'JavaScript programming', 0, 0),
(20, 'post_tag', 'Python programming', 0, 0),
(21, 'post_tag', 'Fitness and exercise', 0, 0),
(22, 'post_tag', 'Movies and cinema', 0, 0),
(23, 'post_tag', 'Online learning platforms', 0, 0),
(24, 'post_tag', 'Personal development', 0, 0),
(25, 'post_tag', 'Cooking and recipes', 0, 0);

-- Link posts to tags
INSERT IGNORE INTO wp_term_relationships (object_id, term_taxonomy_id, term_order) VALUES
(10, 11, 0), (10, 13, 0), (10, 20, 0), -- Machine Learning -> Programming, Data Science, Python
(11, 11, 0), (11, 12, 0), (11, 19, 0), -- React -> Programming, Web Development, JavaScript
(12, 17, 0), -- Climate Change -> Sustainability
(13, 18, 0), -- Digital Marketing -> Digital Marketing
(14, 16, 0), -- Mental Health -> Mindfulness
(15, 15, 0), (15, 25, 0), -- Mediterranean -> Nutrition, Cooking
(16, 21, 0), -- Fitness -> Fitness
(17, 22, 0), -- Cinema -> Movies
(18, 23, 0), -- Online Education -> Online Learning
(19, 24, 0); -- Personal Development -> Self Improvement

-- Insert more sample comments
INSERT IGNORE INTO wp_comments (comment_post_ID, comment_author, comment_author_email, comment_author_url, comment_author_IP, comment_date, comment_date_gmt, comment_content, comment_karma, comment_approved, comment_agent, comment_type, comment_parent, user_id) VALUES
(10, 'Sarah Johnson', 'sarah@example.com', '', '127.0.0.1', NOW(), NOW(), 'Excellent introduction to machine learning! The examples really helped me understand the concepts.', 0, '1', 'Mozilla/5.0', 'comment', 0, 0),
(10, 'Mike Chen', 'mike@example.com', '', '127.0.0.1', NOW(), NOW(), 'As someone new to ML, this article provided a perfect starting point. Looking forward to more advanced topics!', 0, '1', 'Mozilla/5.0', 'comment', 0, 0),
(11, 'Emily Davis', 'emily@example.com', '', '127.0.0.1', NOW(), NOW(), 'React has completely changed how I approach frontend development. Great tutorial!', 0, '1', 'Mozilla/5.0', 'comment', 0, 0),
(11, 'Alex Rodriguez', 'alex@example.com', '', '127.0.0.1', NOW(), NOW(), 'The component lifecycle explanation was particularly helpful. Thanks for sharing!', 0, '1', 'Mozilla/5.0', 'comment', 0, 0),
(12, 'Lisa Wang', 'lisa@example.com', '', '127.0.0.1', NOW(), NOW(), 'Climate change is such a critical issue. Appreciate the practical solutions you suggested.', 0, '1', 'Mozilla/5.0', 'comment', 0, 0),
(13, 'Tom Anderson', 'tom@example.com', '', '127.0.0.1', NOW(), NOW(), 'Digital marketing strategies are evolving so quickly. This article helped me stay current!', 0, '1', 'Mozilla/5.0', 'comment', 0, 0),
(14, 'Rachel Green', 'rachel@example.com', '', '127.0.0.1', NOW(), NOW(), 'Mental health awareness is so important. Thank you for addressing this topic with sensitivity.', 0, '1', 'Mozilla/5.0', 'comment', 0, 0),
(15, 'Carlos Martinez', 'carlos@example.com', '', '127.0.0.1', NOW(), NOW(), 'Mediterranean cuisine is amazing! Tried the recipe suggestions and they were delicious.', 0, '1', 'Mozilla/5.0', 'comment', 0, 0),
(16, 'Jessica Taylor', 'jessica@example.com', '', '127.0.0.1', NOW(), NOW(), 'The fitness guide motivated me to start a new workout routine. Seeing great results!', 0, '1', 'Mozilla/5.0', 'comment', 0, 0),
(17, 'David Kim', 'david@example.com', '', '127.0.0.1', NOW(), NOW(), 'As a film enthusiast, I really enjoyed this analysis of industry trends.', 0, '1', 'Mozilla/5.0', 'comment', 0, 0),
(18, 'Sophie Brown', 'sophie@example.com', '', '127.0.0.1', NOW(), NOW(), 'Online education has opened up so many opportunities. Great insights on the future of learning!', 0, '1', 'Mozilla/5.0', 'comment', 0, 0),
(19, 'Jason Lee', 'jason@example.com', '', '127.0.0.1', NOW(), NOW(), 'Personal development is a journey, not a destination. Thanks for the practical advice!', 0, '1', 'Mozilla/5.0', 'comment', 0, 0);

-- Update comment counts for new posts
UPDATE wp_posts SET comment_count = 2 WHERE ID = 10;
UPDATE wp_posts SET comment_count = 2 WHERE ID = 11;
UPDATE wp_posts SET comment_count = 1 WHERE ID = 12;
UPDATE wp_posts SET comment_count = 1 WHERE ID = 13;
UPDATE wp_posts SET comment_count = 1 WHERE ID = 14;
UPDATE wp_posts SET comment_count = 1 WHERE ID = 15;
UPDATE wp_posts SET comment_count = 1 WHERE ID = 16;
UPDATE wp_posts SET comment_count = 1 WHERE ID = 17;
UPDATE wp_posts SET comment_count = 1 WHERE ID = 18;
UPDATE wp_posts SET comment_count = 1 WHERE ID = 19;

-- Update category counts
UPDATE wp_term_taxonomy SET count = 2 WHERE term_id = 1; -- Technology
UPDATE wp_term_taxonomy SET count = 1 WHERE term_id = 2; -- Science
UPDATE wp_term_taxonomy SET count = 1 WHERE term_id = 3; -- Business
UPDATE wp_term_taxonomy SET count = 1 WHERE term_id = 4; -- Health
UPDATE wp_term_taxonomy SET count = 1 WHERE term_id = 6; -- Food
UPDATE wp_term_taxonomy SET count = 1 WHERE term_id = 7; -- Sports
UPDATE wp_term_taxonomy SET count = 1 WHERE term_id = 8; -- Entertainment
UPDATE wp_term_taxonomy SET count = 1 WHERE term_id = 9; -- Education
UPDATE wp_term_taxonomy SET count = 1 WHERE term_id = 10; -- Lifestyle

-- Update site options
UPDATE wp_options SET option_value = 'My Comprehensive Knowledge Platform' WHERE option_name = 'blogname';
UPDATE wp_options SET option_value = 'Your source for technology, science, business, health, and lifestyle insights' WHERE option_name = 'blogdescription';

SELECT 'Rich WordPress Demo Data Insertion Complete!' as status;
SELECT COUNT(*) as total_posts FROM wp_posts WHERE post_type = 'post' AND post_status = 'publish';
SELECT COUNT(*) as total_pages FROM wp_posts WHERE post_type = 'page' AND post_status = 'publish';
SELECT COUNT(*) as total_comments FROM wp_comments WHERE comment_approved = '1';
SELECT COUNT(*) as total_categories FROM wp_term_taxonomy WHERE taxonomy = 'category';
SELECT COUNT(*) as total_tags FROM wp_term_taxonomy WHERE taxonomy = 'post_tag';


