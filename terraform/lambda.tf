# Lambda

resource "aws_iam_role" "iam_for_lambda" {
  name = "iam_for_lambda"

  assume_role_policy = <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Action": "sts:AssumeRole",
      "Principal": {
        "Service": "lambda.amazonaws.com"
      },
      "Effect": "Allow",
      "Sid": ""
    }
  ]
}
EOF

  tags = {
    Organization = "RTU"
  }
}

resource "aws_lambda_function" "convert_s3_raw_image" {
  count = var.s3_enabled ? 1 : 0
  depends_on = [
    aws_ecr_repository.dataset_image_converter[0]
  ]

  function_name = "convert_s3_raw_image"
  role          = aws_iam_role.iam_for_lambda.arn
  package_type  = "Image"
  image_uri     = "${aws_ecr_repository.dataset_image_converter[0].repository_url}:latest"
  memory_size   = 512
  timeout       = 900

  tags = {
    Organization = "RTU"
  }
}

resource "aws_lambda_permission" "convert_s3_raw_image_s3_perm" {
  count = var.s3_enabled ? 1 : 0
  depends_on = [
    aws_lambda_function.convert_s3_raw_image
  ]
  statement_id  = "AllowExecutionFromS3Bucket"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.convert_s3_raw_image[0].arn
  principal     = "s3.amazonaws.com"
  source_arn    = aws_s3_bucket.rtu_dataset_transport_raw[0].arn
}

resource "aws_s3_bucket_notification" "convert_s3_raw_image_s3_notification" {
  count = var.s3_enabled ? 1 : 0
  depends_on = [
    aws_lambda_permission.convert_s3_raw_image_s3_perm
  ]
  bucket = aws_s3_bucket.rtu_dataset_transport_raw[0].id

  lambda_function {
    lambda_function_arn = aws_lambda_function.convert_s3_raw_image[0].arn
    events              = ["s3:ObjectCreated:*"]
    filter_suffix       = ".ARW"
  }
}
