import org.apache.spark._
import org.apache.spark.sql._
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.SparkContext._
import org.apache.log4j._
import org.apache.spark.util.collection.PairsWriter

object MovieRecommendations {
  
  case class MovieRating(userID: Int, movieID: Int, rating: Float)
  case class MovieName(movieID: Int, movieName: String)
  case class MoviePairs(movie1: Int, movie2: Int, rating1: Float, rating2: Float)
  case class MovieScore(movie1: Int, movie2: Int, score: Float, count: Int)
  
  def parseMovieRating(lines: String): MovieRating = {
    val row = lines.split("\t")
    return MovieRating(row(0).toInt, row(1).toInt, row(2).toFloat)
  }
  
  def parseMovieName(lines: String): MovieName = {
    val row = lines.split('|')
    return MovieName(row(0).toInt, row(1))
  }
  
  def calculateCosineSimilarity(spark: SparkSession, data: Dataset[MoviePairs]): Dataset[MovieScore] = {    
    val pairs = data
                  .withColumn("r1", col("rating1") * col("rating1"))
                  .withColumn("r2", col("rating2") * col("rating2"))
                  .withColumn("r1_r2", col("rating1") * col("rating2"))
    val similarity = pairs
                      .groupBy("movie1", "movie2")
                      .agg(
                          sum(col("r1_r2")).alias("cross_product"),
                          ((sqrt(sum(col("r1")))) * (sqrt(sum(col("r2"))))).alias("vector_lengths"),
                          count(col("r1_r2")).alias("count")
                      )
    
    import spark.implicits._ 
    val result = similarity
                  .withColumn("score",
                      when(col("vector_lengths") =!= 0, col("cross_product")/col("vector_lengths")).otherwise(null))
                  .withColumn("score", col("score").cast("Float"))
                  .withColumn("count", col("count").cast("Int"))
                  .select("movie1", "movie2", "score", "count")
    (return result.as[MovieScore])
  }
  
  def printSimilarMovies(spark: SparkSession, movieSimilarity: Dataset[MovieScore], movieName: Dataset[MovieName], id: Int) {
    movieSimilarity.createOrReplaceTempView("movie_similarity")
    movieName.createOrReplaceTempView("movie_name")
    val movie_name = movieName.filter(col("movieID") === id).select(col("movieName")).first().getString(0)
    val similarMovies = spark.sql(f"""
                                  with cte as
                                  (select t1.movie1, t1.movie2, t2.movieName movieName1, t3.movieName movieName2, t1.score, t1.count
                                  from movie_similarity t1
                                  join movie_name t2
                                    on t1.movie1 = t2.movieID
                                  join movie_name t3
                                    on t1.movie2 = t3.movieID
                                  where (t1.movie1 = ${id} or t1.movie2 = ${id})
                                  and t1.count >= 50)
                                  select case when movie1 = ${id} then movieName2 else movieName1 end movie, score, count
                                  from cte
                                  order by score desc
                                  limit 20
                                  """)
     if (similarMovies.rdd.isEmpty()) {
       println(f"There Are No Movies We Can Recommend for ${movie_name}.")
     } else {
       val number_of_similar_movies = similarMovies.rdd.count()
       var count = 1
       println(f"Here Are The Top ${number_of_similar_movies} Movies We Recommend After Watching ${movie_name}: ")
       for (row <- similarMovies.rdd.collect()){
         val similarMovie = row(0)
         val similarityScore = row(1)
         val numberOfVotes = row(2)
         //println(f"- ${count}. ${similarMovie} with a Similarity Score of ${similarityScore} From ${numberOfVotes} Votes")
         println(f"- ${count}. ${similarMovie}")
         count += 1
       }
     }
  }
  
  def main(args: Array[String]) {
    
    // Set the log level to only print the errors
    //Logger.getLogger("org").setLevel(Level.FATAL)
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)
    
    // Create a SparkContext using every core of the local machine, named MovieRecommendations
    val sc = new SparkContext("local[*]", "MovieRecommendations")
    
     // Use SparkSession interface; need to implement spark.stop() at the end of the script to end session on the cluster
    val spark = SparkSession
      .builder
      .appName("MovieRecommendations") 
      .master("local[*]") // run on local machine
      .getOrCreate()
      
    import spark.implicits._
    val movieRating = sc.textFile("movieRatings.data")
    val movieRatingParsed = movieRating.map(parseMovieRating)
    val movieRatingDS = movieRatingParsed.toDS
    val movieName = sc.textFile("movieNames.item")
    val movieNameParsed = movieName.map(parseMovieName)
    val movieNameDS = movieNameParsed.toDS

    val movieRatingSelfPairs = movieRatingDS.as("t1")
                            .join(movieRatingDS.as("t2"), $"t1.userID" === $"t2.userID" && $"t1.movieID" < $"t2.movieID")
                            .select(col("t1.movieID").alias("movie1"), 
                                    col("t2.movieID").alias("movie2"), 
                                    col("t1.rating").alias("rating1"),
                                    col("t2.rating").alias("rating2"))   
    
    val movieRatingSelfPairsFiltered = movieRatingSelfPairs.filter(col("rating1") > 2 && col("rating2") > 2)
    val similarityDS = calculateCosineSimilarity(spark, movieRatingSelfPairsFiltered.as[MoviePairs])
    val movieID = 1 // corresponds to Toy Story
    printSimilarMovies(spark, similarityDS, movieNameDS, movieID)
    spark.stop()
  }
  
}